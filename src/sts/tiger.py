from RealtimeSTT import AudioToTextRecorder
from kittentts import KittenTTS
import soundfile as sf
import random
import os
import time

class TigerAI:
    def __init__(self):
        self.tts = KittenTTS("KittenML/kitten-tts-nano-0.1")
        self.recorder = AudioToTextRecorder()
        self.conversation_text = ""
        self.is_speaking = False
        self.jokes = [
            "Why don't scientists trust atoms? Because they make up everything! ",
            "I told my wife she was drawing her eyebrows too high. She looked surprised! ",
            "Why don't eggs tell jokes? They'd crack each other up! ",
            "What do you call a fake noodle? An impasta! ",
            "Why did the scarecrow win an award? He was outstanding in his field! "
        ]
    
    def llm(self, prompt):
        joke = random.choice(self.jokes)
        return joke + prompt
    
    def process_speech(self, text):
        if self.is_speaking:
            return
            
        self.conversation_text += text + " "
        print(f"You said: {text}")
        
        if "tiger" in text.lower():
            print("Processing your input...")
            self.is_speaking = True
            response = self.llm(self.conversation_text.replace("tiger", "").strip())
            self.speak(response)
            self.conversation_text = ""
            self.is_speaking = False
    
    def speak(self, text):
        print(f"AI responds: {text}")
        audio = self.tts.generate(text, voice='expr-voice-2-f', speed=1.45)
        sf.write('tiger_output.wav', audio, 24000)
        
        os.system('aplay tiger_output.wav 2>/dev/null || paplay tiger_output.wav 2>/dev/null || afplay tiger_output.wav 2>/dev/null')
    
if __name__ == '__main__':
    tiger = TigerAI()
    
    while True:
        try:
            tiger.recorder.start()
            input("Press Enter to stop recording...")
            tiger.recorder.stop()
            
            text = tiger.recorder.text()
            print("Transcription: ", text)
            
            if "tiger" in text.lower():
                print("Processing your input...")
                response = tiger.llm(text.replace("tiger", "").strip())
                tiger.speak(response)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
