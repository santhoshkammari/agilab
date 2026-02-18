#!/usr/bin/env python3
"""Jarvis Voice Assistant - Always-listening voice assistant.

Pipeline: Wake word ("Jarvis") -> Record speech -> Transcribe (Parakeet V3 ONNX)
          -> AI reasoning (Qwen CLI) -> Speak response (Pocket TTS) -> Loop
"""

import sys
import json
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd

# ─── Configuration ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # For wake word + STT
TTS_SAMPLE_RATE = 24000      # Pocket TTS output rate
WAKE_CHUNK_MS = 80           # openwakeword needs 80ms chunks
WAKE_CHUNK_SAMPLES = int(SAMPLE_RATE * WAKE_CHUNK_MS / 1000)  # 1280
WAKE_THRESHOLD = 0.5         # Wake word confidence threshold

SILENCE_THRESHOLD = 0.015    # RMS energy below this = silence
SILENCE_DURATION = 1.5       # Seconds of silence to stop recording
MAX_RECORD_SECONDS = 30      # Max recording length
MIN_RECORD_SECONDS = 0.5     # Minimum speech to process

PARAKEET_MODEL_DIR = Path.home() / ".local/share/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8"

QWEN_AGENT_PATH = Path.home() / "buildmode/agilab/src/mcp_tools/qwen_agent.py"
INCLUDE_DIRECTORIES = ['/home', '/tmp', '/etc', '/var', '/opt']

TTS_VOICE = "alba"  # Built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma

BLANK_TOKEN = 8192
VOCAB_SIZE = 8193           # 0..8192 inclusive (8192 = blank)
NUM_DURATIONS = 5           # TDT durations: 0,1,2,3,4
PRED_RNN_LAYERS = 2
PRED_HIDDEN = 640


# ─── Parakeet V3 ONNX Speech-to-Text ────────────────────────────────────────
class ParakeetSTT:
    """Parakeet TDT 0.6B v3 INT8 ONNX inference."""

    def __init__(self, model_dir: Path):
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

        # Load vocabulary
        self.vocab = []
        with open(model_dir / "vocab.txt") as f:
            for line in f:
                token = line.rsplit(" ", 1)[0]
                self.vocab.append(token)

        print(f"[STT] Loaded. Vocab size: {len(self.vocab)}")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio (float32, 16kHz, mono) to text."""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add batch dim

        audio = audio.astype(np.float32)
        audio_len = np.array([audio.shape[1]], dtype=np.int64)

        # Extract mel features
        features, feat_lens = self.feature_extractor.run(
            None, {"waveforms": audio, "waveforms_lens": audio_len}
        )
        # features: [batch, 128, T]

        # Encode
        encoded, enc_lens = self.encoder.run(
            None, {"audio_signal": features, "length": feat_lens}
        )
        # encoded: [batch, 1024, T_enc]

        # TDT greedy decode
        return self._tdt_greedy_decode(encoded, enc_lens)

    def _tdt_greedy_decode(self, encoded: np.ndarray, enc_lens: np.ndarray) -> str:
        """TDT (Token-and-Duration Transducer) greedy decoding."""
        T = enc_lens[0]
        # Initialize prediction network states
        state1 = np.zeros((PRED_RNN_LAYERS, 1, PRED_HIDDEN), dtype=np.float32)
        state2 = np.zeros((PRED_RNN_LAYERS, 1, PRED_HIDDEN), dtype=np.float32)

        # Start with blank token as input
        targets = np.array([[BLANK_TOKEN]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)

        tokens = []
        t = 0

        while t < T:
            # Get encoder output at time t
            enc_t = encoded[:, :, t:t+1]  # [1, 1024, 1]

            # Run decoder/joint
            outputs, _, state1, state2 = self.decoder.run(
                None, {
                    "encoder_outputs": enc_t,
                    "targets": targets,
                    "target_length": target_length,
                    "input_states_1": state1,
                    "input_states_2": state2,
                }
            )
            # outputs: [1, 1, 1, vocab_size + num_durations]
            logits = outputs[0, 0, 0]  # [8198]

            # Split into vocab logits and duration logits
            vocab_logits = logits[:VOCAB_SIZE]
            duration_logits = logits[VOCAB_SIZE:]

            token_id = int(np.argmax(vocab_logits))
            duration = int(np.argmax(duration_logits))

            if token_id != BLANK_TOKEN:
                tokens.append(token_id)
                targets = np.array([[token_id]], dtype=np.int32)

            # Advance by at least 1, plus duration
            t += max(1, duration)

        # Decode tokens to text
        text = self._decode_tokens(tokens)
        return text

    def _decode_tokens(self, token_ids: list) -> str:
        """Convert token IDs to text using SentencePiece-style decoding."""
        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.vocab):
                piece = self.vocab[tid]
                # Skip special tokens
                if piece.startswith("<") and piece.endswith(">"):
                    continue
                pieces.append(piece)

        text = "".join(pieces)
        # SentencePiece uses ▁ (U+2581) for word boundaries
        text = text.replace("▁", " ").strip()
        return text


# ─── Wake Word Detection ────────────────────────────────────────────────────
class WakeWordDetector:
    """OpenWakeWord-based wake word detection for 'hey jarvis'."""

    def __init__(self):
        print("[WAKE] Loading openwakeword model...")
        from openwakeword.model import Model
        self.model = Model()
        # We use the 'hey_jarvis' model
        print(f"[WAKE] Available models: {list(self.model.models.keys())}")

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process 80ms audio chunk. Returns True if wake word detected."""
        # openwakeword expects int16 audio
        if audio_chunk.dtype == np.float32:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        else:
            audio_int16 = audio_chunk.astype(np.int16)

        prediction = self.model.predict(audio_int16)
        score = prediction.get("hey_jarvis", 0)
        if score > WAKE_THRESHOLD:
            self.model.reset()  # Reset to avoid repeated triggers
            return True
        return False


# ─── Speech Recorder ────────────────────────────────────────────────────────
class SpeechRecorder:
    """Records speech from microphone until silence is detected."""

    def record(self) -> np.ndarray | None:
        """Record speech, return float32 16kHz mono numpy array or None."""
        print("[REC] Recording... (speak now)")
        chunks = []
        silence_samples = 0
        max_samples = int(MAX_RECORD_SECONDS * SAMPLE_RATE)
        silence_limit = int(SILENCE_DURATION * SAMPLE_RATE)
        chunk_size = 1600  # 100ms chunks

        total_samples = 0
        speech_started = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk_size) as stream:
            while total_samples < max_samples:
                data, _ = stream.read(chunk_size)
                audio = data[:, 0]  # mono
                chunks.append(audio)
                total_samples += len(audio)

                rms = np.sqrt(np.mean(audio ** 2))

                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    silence_samples = 0
                elif speech_started:
                    silence_samples += len(audio)
                    if silence_samples >= silence_limit:
                        print("[REC] Silence detected, stopping.")
                        break

        recording = np.concatenate(chunks)
        duration = len(recording) / SAMPLE_RATE

        if duration < MIN_RECORD_SECONDS:
            print(f"[REC] Too short ({duration:.1f}s), ignoring.")
            return None

        print(f"[REC] Recorded {duration:.1f}s of audio.")
        return recording


# ─── Qwen AI Processing ─────────────────────────────────────────────────────
def qwen_process(prompt: str) -> str:
    """Send prompt to Qwen CLI and return the text response."""
    print(f"[AI] Processing: {prompt[:80]}...")

    cmd = ['qwen', '-y', '--output-format', 'stream-json', '-p', prompt]
    for d in INCLUDE_DIRECTORIES:
        cmd.extend(['--include-directories', d])

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1
        )

        result_text = ""
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if chunk.get('type') == 'result':
                    result_text = chunk.get('result', '')
                elif chunk.get('type') == 'assistant' and 'message' in chunk:
                    msg = chunk['message']
                    if isinstance(msg, dict) and msg.get('content'):
                        content = msg['content']
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    result_text = block.get('text', '')
                        elif isinstance(content, str):
                            result_text = content
            except json.JSONDecodeError:
                continue

        process.stdout.close()
        process.wait(timeout=120)

        if result_text:
            print(f"[AI] Response: {result_text[:100]}...")
            return result_text
        return "I'm sorry, I couldn't process that."

    except Exception as e:
        print(f"[AI] Error: {e}")
        return "Sorry, there was an error processing your request."


# ─── Text-to-Speech ─────────────────────────────────────────────────────────
class Speaker:
    """Pocket TTS text-to-speech."""

    def __init__(self):
        print("[TTS] Loading Pocket TTS model...")
        from pocket_tts import TTSModel
        self.model = TTSModel.load_model()
        print("[TTS] Loading voice state...")
        self.voice_state = self.model.get_state_for_audio_prompt(TTS_VOICE)
        self.sample_rate = self.model.sample_rate
        print(f"[TTS] Ready. Sample rate: {self.sample_rate}")

    def speak(self, text: str):
        """Generate and play speech from text."""
        if not text.strip():
            return

        # Truncate very long responses for TTS
        if len(text) > 500:
            text = text[:500] + "..."

        print(f"[TTS] Speaking: {text[:80]}...")
        try:
            audio_tensor = self.model.generate_audio(
                self.voice_state, text, copy_state=True
            )
            audio_np = audio_tensor.squeeze().cpu().numpy()

            # Play audio (blocking)
            sd.play(audio_np, samplerate=self.sample_rate)
            sd.wait()
            print("[TTS] Done speaking.")
        except Exception as e:
            print(f"[TTS] Error: {e}")


# ─── Main Assistant Loop ────────────────────────────────────────────────────
class Jarvis:
    """Main Jarvis voice assistant."""

    def __init__(self):
        print("=" * 60)
        print("  JARVIS Voice Assistant")
        print("=" * 60)
        print()

        self.wake = WakeWordDetector()
        self.stt = ParakeetSTT(PARAKEET_MODEL_DIR)
        self.recorder = SpeechRecorder()
        self.speaker = Speaker()
        self._muted = False

        print()
        print("[JARVIS] Ready! Say 'Hey Jarvis' to activate.")
        print("[JARVIS] Press Ctrl+C to exit.")
        print()

    def run(self):
        """Main loop: listen for wake word, then process."""
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=WAKE_CHUNK_SAMPLES) as stream:
                while True:
                    if self._muted:
                        time.sleep(0.1)
                        continue

                    data, _ = stream.read(WAKE_CHUNK_SAMPLES)
                    audio = data[:, 0]

                    if self.wake.process_chunk(audio):
                        print("\n[JARVIS] Wake word detected!")
                        self._handle_interaction()

        except KeyboardInterrupt:
            print("\n[JARVIS] Goodbye!")

    def _handle_interaction(self):
        """Handle one full interaction: record -> transcribe -> AI -> speak."""
        # Play a short beep to indicate listening
        self._play_beep()

        # Record speech
        audio = self.recorder.record()
        if audio is None:
            return

        # Transcribe
        print("[JARVIS] Transcribing...")
        text = self.stt.transcribe(audio)
        print(f"[JARVIS] You said: '{text}'")

        if not text.strip():
            print("[JARVIS] Empty transcription, ignoring.")
            return

        # AI processing
        response = qwen_process(text)

        # Speak response (mute mic during playback)
        self._muted = True
        try:
            self.speaker.speak(response)
        finally:
            self._muted = False

    def _play_beep(self):
        """Play a short beep to indicate wake word detected."""
        duration = 0.15
        freq = 880
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
        beep = 0.3 * np.sin(2 * np.pi * freq * t)
        # Fade in/out
        fade = int(SAMPLE_RATE * 0.02)
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        sd.play(beep, samplerate=SAMPLE_RATE)
        sd.wait()


if __name__ == "__main__":
    jarvis = Jarvis()
    jarvis.run()
