#!/usr/bin/env python3
"""Jarvis Voice Assistant - Always-listening voice assistant.

Pipeline: Wake word ("Jarvis") -> Record speech -> Transcribe (Parakeet V3 ONNX)
          -> AI reasoning (Qwen CLI) -> Speak response (KittenTTS) -> Loop
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    # Audio
    sample_rate: int = 16000

    # Wake word
    wake_chunk_ms: int = 80
    wake_threshold: float = 0.5

    # Recording
    silence_threshold: float = 0.005  # RMS below this = silence
    silence_duration: float = 1.5     # seconds of silence to stop
    max_record_seconds: float = 30.0
    min_record_seconds: float = 0.5
    speech_start_timeout: float = 8.0  # give up if no speech detected within this

    # STT model
    parakeet_model_dir: Path = field(
        default_factory=lambda: Path.home() / ".local/share/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8"
    )

    # Parakeet constants
    blank_token: int = 8192
    vocab_size: int = 8193
    pred_rnn_layers: int = 2
    pred_hidden: int = 640

    # Qwen AI
    include_directories: list = field(default_factory=lambda: ['/home', '/tmp', '/etc', '/var', '/opt'])

    # TTS
    kitten_voice: str = "expr-voice-5-m"
    kitten_speed: float = 1.5
    tts_max_chars: int = 500

    @property
    def wake_chunk_samples(self) -> int:
        return int(self.sample_rate * self.wake_chunk_ms / 1000)


CFG = Config()


# ─── Parakeet V3 ONNX Speech-to-Text ────────────────────────────────────────

class ParakeetSTT:
    """Parakeet TDT 0.6B v3 INT8 ONNX inference."""

    def __init__(self, model_dir: Path = CFG.parakeet_model_dir):
        print("[STT] Loading Parakeet V3 ONNX models...")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4

        self.feature_extractor = ort.InferenceSession(
            str(model_dir / "nemo128.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.encoder = ort.InferenceSession(
            str(model_dir / "encoder-model.int8.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(model_dir / "decoder_joint-model.int8.onnx"), opts, providers=["CPUExecutionProvider"]
        )

        self.vocab = []
        with open(model_dir / "vocab.txt") as f:
            for line in f:
                self.vocab.append(line.rsplit(" ", 1)[0])

        print(f"[STT] Loaded. Vocab size: {len(self.vocab)}")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = audio.astype(np.float32)
        audio_len = np.array([audio.shape[1]], dtype=np.int64)

        features, feat_lens = self.feature_extractor.run(
            None, {"waveforms": audio, "waveforms_lens": audio_len}
        )
        encoded, enc_lens = self.encoder.run(
            None, {"audio_signal": features, "length": feat_lens}
        )
        return self._tdt_greedy_decode(encoded, enc_lens)

    def _tdt_greedy_decode(self, encoded: np.ndarray, enc_lens: np.ndarray) -> str:
        T = enc_lens[0]
        state1 = np.zeros((CFG.pred_rnn_layers, 1, CFG.pred_hidden), dtype=np.float32)
        state2 = np.zeros((CFG.pred_rnn_layers, 1, CFG.pred_hidden), dtype=np.float32)
        targets = np.array([[CFG.blank_token]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)

        tokens = []
        t = 0
        while t < T:
            enc_t = encoded[:, :, t:t+1]
            outputs, _, state1, state2 = self.decoder.run(
                None, {
                    "encoder_outputs": enc_t,
                    "targets": targets,
                    "target_length": target_length,
                    "input_states_1": state1,
                    "input_states_2": state2,
                }
            )
            logits = outputs[0, 0, 0]
            token_id = int(np.argmax(logits[:CFG.vocab_size]))
            duration = int(np.argmax(logits[CFG.vocab_size:]))

            if token_id != CFG.blank_token:
                tokens.append(token_id)
                targets = np.array([[token_id]], dtype=np.int32)

            t += max(1, duration)

        return self._decode_tokens(tokens)

    def _decode_tokens(self, token_ids: list) -> str:
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.vocab):
                piece = self.vocab[tid]
                if not (piece.startswith("<") and piece.endswith(">")):
                    pieces.append(piece)
        return "".join(pieces).replace("▁", " ").strip()


# ─── Wake Word Detection ─────────────────────────────────────────────────────

class WakeWordDetector:
    """OpenWakeWord-based wake word detection for 'alexa'."""

    def __init__(self):
        print("[WAKE] Loading openwakeword model...")
        from openwakeword.model import Model
        self.model = Model()
        print(f"[WAKE] Available models: {list(self.model.models.keys())}")

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        if audio_chunk.dtype == np.float32:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk.astype(np.int16)

        prediction = self.model.predict(audio_int16)
        score = prediction.get("alexa", 0)
        if score > CFG.wake_threshold:
            self.model.reset()
            return True
        return False


# ─── Speech Recorder ─────────────────────────────────────────────────────────

class SpeechRecorder:
    """Records speech from microphone until silence is detected."""

    def record(self) -> np.ndarray | None:
        """Record from mic immediately, stop on silence. No threshold gating."""
        print("[REC] Recording... (speak now)")
        chunk_size = 1600  # 100ms
        max_samples = int(CFG.max_record_seconds * CFG.sample_rate)
        silence_limit = int(CFG.silence_duration * CFG.sample_rate)
        start_timeout = int(CFG.speech_start_timeout * CFG.sample_rate)

        chunks = []
        silence_samples = 0
        speech_started = False
        waiting_samples = 0
        total_samples = 0

        with sd.InputStream(samplerate=CFG.sample_rate, channels=1, dtype="float32",
                            blocksize=chunk_size) as stream:
            # Flush a couple chunks for stream to stabilize
            for _ in range(2):
                stream.read(chunk_size)

            while total_samples < max_samples:
                data, _ = stream.read(chunk_size)
                audio = data[:, 0]
                chunks.append(audio)
                total_samples += len(audio)

                rms = float(np.sqrt(np.mean(audio ** 2)))

                if rms > CFG.silence_threshold:
                    speech_started = True
                    silence_samples = 0
                elif speech_started:
                    # Count trailing silence only after speech has begun
                    silence_samples += len(audio)
                    if silence_samples >= silence_limit:
                        print("[REC] Silence detected, stopping.")
                        break
                else:
                    # Still waiting for speech — timeout if too long
                    waiting_samples += len(audio)
                    if waiting_samples >= start_timeout:
                        print("[REC] No speech detected, giving up.")
                        return None

        if not speech_started:
            print("[REC] No speech captured.")
            return None

        recording = np.concatenate(chunks)
        duration = len(recording) / CFG.sample_rate
        print(f"[REC] Recorded {duration:.1f}s of audio.")
        return recording


# ─── AI Processor ────────────────────────────────────────────────────────────

class AIProcessor:
    """Wraps Qwen CLI. Returns full text response."""

    def __init__(self):
        self.include_dirs = CFG.include_directories

    def process(self, prompt: str) -> str:
        print(f"[AI] Processing: {prompt[:80]}...")
        cmd = ['qwen', '-y', '--output-format', 'stream-json', '-p', prompt]
        for d in self.include_dirs:
            cmd.extend(['--include-directories', d])

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, bufsize=1
            )

            result_text = ""
            for line in iter(proc.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    # Use the final 'result' field — the complete response
                    if chunk.get('type') == 'result':
                        result_text = chunk.get('result', '')
                except json.JSONDecodeError:
                    continue

            proc.stdout.close()
            proc.wait(timeout=120)

            if result_text:
                print(f"[AI] Response: {result_text[:100]}...")
                return result_text
            return "I'm sorry, I couldn't process that."

        except Exception as e:
            print(f"[AI] Error: {e}")
            return "Sorry, there was an error processing your request."


# ─── Speaker ─────────────────────────────────────────────────────────────────

class KittenSpeaker:
    """KittenTTS — 15M param, CPU-only TTS.

    Install: pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
    Voices: expr-voice-2-m/f, expr-voice-3-m/f, expr-voice-4-m/f, expr-voice-5-m/f
    """

    SAMPLE_RATE = 24000

    def __init__(self, voice: str = CFG.kitten_voice, speed: float = CFG.kitten_speed):
        print("[TTS] Loading KittenTTS model...")
        from kittentts import KittenTTS
        self.model = KittenTTS("KittenML/kitten-tts-nano-0.2")
        self.voice = voice
        self.speed = speed
        self.sample_rate = self.SAMPLE_RATE
        print(f"[TTS] Ready. Voice: {self.voice}, Speed: {self.speed}x")

    def speak(self, text: str):
        if not text.strip():
            return
        if len(text) > CFG.tts_max_chars:
            text = text[:CFG.tts_max_chars] + "..."
        print(f"[TTS] Speaking: {text[:80]}...")
        try:
            audio = self.model.generate(text, voice=self.voice, speed=self.speed)
            if audio is None or len(audio) == 0:
                print("[TTS] Empty audio returned, skipping.")
                return
            # Pad 0.3s silence at the end — KittenTTS trims aggressively and
            # sd.wait() returns too early on short clips, cutting off the tail
            silence = np.zeros(int(self.sample_rate * 0.3), dtype=audio.dtype)
            audio = np.concatenate([audio, silence])
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
        print("[TTS] Done.")


# ─── Jarvis Assistant ─────────────────────────────────────────────────────────

class JarvisAssistant:
    """Main Jarvis voice assistant. Call .run() to start."""

    def __init__(self):
        print("=" * 60)
        print("  JARVIS Voice Assistant")
        print("=" * 60)

        self.wake = WakeWordDetector()
        self.stt = ParakeetSTT()
        self.recorder = SpeechRecorder()
        self.ai = AIProcessor()
        self.speaker = KittenSpeaker()

        print()
        print("[JARVIS] Ready! Say 'Alexa' to activate.")
        print("[JARVIS] Press Ctrl+C to exit.")

    def run(self):
        """Main loop: listen for wake word, then process."""
        try:
            while True:
                with sd.InputStream(samplerate=CFG.sample_rate, channels=1, dtype="float32",
                                    blocksize=CFG.wake_chunk_samples) as stream:
                    while True:
                        data, _ = stream.read(CFG.wake_chunk_samples)
                        if self.wake.process_chunk(data[:, 0]):
                            print("\n[JARVIS] Wake word detected!")
                            break  # closes wake stream

                self._handle_interaction()

        except KeyboardInterrupt:
            print("\n[JARVIS] Goodbye!")

    def _handle_interaction(self):
        """One full turn: beep -> record -> transcribe -> AI -> speak."""
        self._play_beep()

        audio = self.recorder.record()
        if audio is None:
            return

        print("[JARVIS] Transcribing...")
        text = self.stt.transcribe(audio)
        print(f"[JARVIS] You said: '{text}'")

        if not text.strip():
            print("[JARVIS] Empty transcription, ignoring.")
            return

        response = self.ai.process(text)
        self.speaker.speak(response)

    def _play_beep(self):
        duration, freq = 0.15, 880
        t = np.linspace(0, duration, int(CFG.sample_rate * duration), dtype=np.float32)
        beep = 0.3 * np.sin(2 * np.pi * freq * t)
        fade = int(CFG.sample_rate * 0.02)
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        sd.play(beep, samplerate=CFG.sample_rate)
        sd.wait()


if __name__ == "__main__":
    JarvisAssistant().run()
