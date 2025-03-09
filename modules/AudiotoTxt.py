import sounddevice as sd
import numpy as np
import whisper

# Load Whisper model
model = whisper.load_model("base")  # Use "base" for faster performance

# Parameters
sample_rate = 16000  # Sample rate in Hz (Whisper expects 16 kHz)
duration = 5         # Duration of each recording in seconds

print("Recording... Speak now!")

# Record audio
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()  # Wait until recording is finished

# Transcribe audio using Whisper (English only)
result = model.transcribe(audio_data.flatten(), fp16=False, language="en")  # Set language to English
print("Transcribed Text:", result["text"])