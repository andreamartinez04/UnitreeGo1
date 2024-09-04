import os
import subprocess
import speech_recognition as sr
import numpy as np
import soundfile as sf
import tempfile
from whisper import load_model

# Load the Whisper model
model = load_model("base")

# Define the path for Node.js script
node_script_path = "C:/Users/charl/Desktop/Unitree-Go1-NodeJS-ChatGPT/index.js"

# Start the Node.js process
node_process = subprocess.Popen(['node', node_script_path], stdin=subprocess.PIPE, text=True, bufsize=1)

def transcribe_audio_to_text():
    """Transcribes spoken commands into text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio_data = recognizer.listen(source)
        print("Processing speech...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(audio_data.get_wav_data())
            tmpfile.close()
            # Read the audio data from the file
            audio, samplerate = sf.read(tmpfile.name)
        os.remove(tmpfile.name)
        # Convert audio to numpy array and cast to float32
        audio_np = np.array(audio, dtype=np.float32)
        result = model.transcribe(audio_np)
        return result["text"].lower()

def send_command_to_node(command):
    """Sends command to the Node.js script."""
    print(f"Sending command to Node.js: {command}")
    try:
        node_process.stdin.write(command + "\n")
        node_process.stdin.flush()
    except Exception as e:
        print(f"Failed to send command: {e}")

if __name__ == "__main__":
    try:
        while True:
            command = transcribe_audio_to_text()
            if command:
                send_command_to_node(command)
            else:
                print("No command detected.")
    except KeyboardInterrupt:
        print("Shutting down...")
        node_process.terminate()
