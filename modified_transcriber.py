import sounddevice as sd
import numpy as np
import queue
import whisper
import webrtcvad
import torch
import wave
import time
import threading

# Load Whisper model
whisper_model = whisper.load_model("base")

# WebRTC VAD setup
vad = webrtcvad.Vad(3)  # Mode 3 is aggressive mode, works well for real-time use.

# Queue for audio chunks
caller_queue = queue.Queue()
operator_queue = queue.Queue()

# Audio settings
SAMPLE_RATE = 16000  # Whisper works best with 16kHz audio
CHANNELS = 1
CHUNK_SIZE = 1024  # Adjust for latency vs. processing
FRAME_DURATION = 30  # in ms for each frame processed by VAD

def callback_caller(indata, frames, time, status):
    """Callback function to store caller microphone audio in queue."""
    if status:
        print(f"Caller recording error: {status}", flush=True)
    caller_queue.put(indata.copy())

def callback_operator(indata, frames, time, status):
    """Callback function to store operator microphone audio in queue."""
    if status:
        print(f"Operator recording error: {status}", flush=True)
    operator_queue.put(indata.copy())

def process_audio(queue, label):
    """Processes live audio using VAD and transcribes with Whisper."""
    print(f"Listening for {label} speech...")

    frames_buffer = []

    while True:
        # Get the latest chunk of audio data
        audio_chunk = queue.get()
        frames_buffer.extend(audio_chunk)

        # Frame size in samples (30ms frames)
        frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000)

        while len(frames_buffer) >= frame_size:
            # Extract a 30ms chunk for VAD
            frame = frames_buffer[:frame_size]
            frames_buffer = frames_buffer[frame_size:]

            # Check if there's speech in this frame using WebRTC VAD
            is_speech = vad.is_speech(np.array(frame), SAMPLE_RATE)

            if is_speech:
                # Save temp file for processing with Whisper
                temp_filename = f"{label}_live_audio.wav"
                with wave.open(temp_filename, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(np.array(frame).tobytes())

                # Use Whisper to transcribe the speech
                result = whisper_model.transcribe(temp_filename)
                print(f"{label} Transcription: {result['text']}")

            time.sleep(0.01)  # Small sleep to prevent maxing out CPU

# Run separate threads for caller and operator
def start_streams():
    # Create threads for both caller and operator audio streams
    caller_thread = threading.Thread(target=process_audio, args=(caller_queue, "Caller"))
    operator_thread = threading.Thread(target=process_audio, args=(operator_queue, "Operator"))

    # Start both threads
    caller_thread.start()
    operator_thread.start()

    with sd.InputStream(callback=callback_caller, channels=CHANNELS, samplerate=SAMPLE_RATE):
        with sd.InputStream(callback=callback_operator, channels=CHANNELS, samplerate=SAMPLE_RATE):
            print("Listening to both caller and operator...")
            while True:
                # Main thread will continue running, waiting for audio from both streams
                time.sleep(1)

# Start the audio streams and processing
start_streams()
