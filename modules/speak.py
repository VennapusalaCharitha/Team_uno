import pyttsx3

def speak_text(text):
    engine = pyttsx3.init()
    
    # Optional: Set voice, rate, and volume
    engine.setProperty('rate', 150)  # Speed
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)
    
    # Speak the text1
    engine.say(text)
    engine.runAndWait()

# usage
speak_text("Hello, how are you?")
