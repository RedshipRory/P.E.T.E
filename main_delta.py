import threading
import sounddevice as sd
import operator_transcriber
import voice_transcriber_caller
import whisper
import webrtcvad
import re
import os
import numpy as np
import scipy.io.wavfile as wav

# Replace with actual device IDs from `sd.query_devices()`
MIC_DEVICE = 1  # Your Microphone Array (AMD Audio Dev)

SAMPLE_RATE = 16000 
CHANNELS = 2

# Define callback function to process incoming audio
def callback(indata, frames, time, status):
    if status:
        print(status)  # Print any stream errors
    print(indata)  # Process or analyze the audio data here

# Function to list available microphones
def list_microphones():
    devices = sd.query_devices()
    mic_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Check if the device can be used for input (microphone)
            mic_devices.append((i, device['name']))
    return mic_devices

# Function to select a microphone by ID
def select_microphone_by_id(mic_id):
    devices = sd.query_devices()
    try:
        # Get the device info for the specified microphone ID
        device_info = devices[mic_id]
        if device_info['max_input_channels'] > 0:
            print(f"Selected Microphone: {device_info['name']}")
            return mic_id  # Return the device ID to be used for capturing audio
        else:
            print("Selected device is not an input device.")
            return None
    except IndexError:
        print(f"Error: Microphone ID {mic_id} does not exist.")
        return None

# Function to handle microphone input and transcription for the operator mic
def handle_microphone_input_operator(mic_id):
    model = whisper.load_model("base")

    # Select the microphone by ID
    mic_id = select_microphone_by_id(mic_id)
    if mic_id is None:
        print(f"Skipping microphone {mic_id} as it is not valid.")
        return  # Skip if the microphone is not valid

    while True:
        try:
            sample_rate = 24000
            agressiveness = 0
            vad = webrtcvad.Vad(int(agressiveness))

            # Assuming vad_collector works as expected for this mic_id
            segments = operator_transcriber.vad_collector(sample_rate, 30, 1500, vad)

            for i, segment in enumerate(segments):
                path = f'chunk-{mic_id}-%002d.wav' % (i,)
                print(f'Writing {path}')
                operator_transcriber.write_wave(path, segment, sample_rate)
                result = operator_transcriber.transcribe_speech(path, model)

                print("Transcription:")
                print(result['text'])

                # Delete the file after processing
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error deleting file {path}: {e}")
        except Exception as e:
            print(f"Error in operator mic transcription: {e}")

# Function to handle microphone input and transcription for the caller mic
def handle_microphone_input_caller(mic_id):
    model = whisper.load_model("base")

    # Select the microphone by ID
    mic_id = select_microphone_by_id(mic_id)
    if mic_id is None:
        print(f"Skipping microphone {mic_id} as it is not valid.")
        return  # Skip if the microphone is not valid

    while True:
        try:
            sample_rate = 24000
            agressiveness = 0
            vad = webrtcvad.Vad(int(agressiveness))

            # Assuming vad_collector works as expected for this mic_id
            segments = voice_transcriber_caller.vad_collector(sample_rate, 30, 1500, vad)

            for i, segment in enumerate(segments):
                path = f'chunk-{mic_id}-%002d.wav' % (i,)
                print(f'Writing {path}')
                voice_transcriber_caller.write_wave(path, segment, sample_rate)
                result = voice_transcriber_caller.transcribe_speech(path, model)

                print("Transcription:")
                print(result['text'])

                # Delete the file after processing
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error deleting file {path}: {e}")
        except Exception as e:
            print(f"Error in caller mic transcription: {e}")

# Function to locate microphone by name
def locateMic(name, mic_devices):
    for mic_id, mic_name in mic_devices:
        match = re.search(name, mic_name)
        if match:
            return mic_id
    return None  # Return None if the microphone was not found

# Main function to start the threads for both microphones
def main():
    # List available microphones
    mic_devices = list_microphones()
    print("Available Microphones:")
    for mic_id, mic_name in mic_devices:
        print(f"ID: {mic_id}, Name: {mic_name}")

    # Locate the microphones by name
    operator_mic = locateMic("NVIDIA Broadcast", mic_devices)
    caller_mic = locateMic("Virtual Audio Cable", mic_devices)

    if operator_mic is None or caller_mic is None:
        print("Error: One or both microphones could not be found.")
        return

    # Starting two threads for two microphones (selecting by ID)
    mic1_thread = threading.Thread(target=handle_microphone_input_operator, args=(mic_devices[operator_mic][0],))  # Using operator mic
    mic2_thread = threading.Thread(target=handle_microphone_input_caller, args=(mic_devices[caller_mic][0],))  # Using caller mic

    mic1_thread.start()
    mic2_thread.start()
    
    # Join threads to wait for their completion
    mic1_thread.join()
    mic2_thread.join()

if __name__ == "__main__":
    main()

