import threading
import sounddevice as sd
import voice_transcriber
import whisper
import webrtcvad
import re

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

# Function to handle microphone input and transcription
def handle_microphone_input(mic_id):
    model = whisper.load_model("base")

    while True:
        sample_rate = 24000
        agressiveness = 0
        vad = webrtcvad.Vad(int(agressiveness))
        segments = voice_transcriber.vad_collector(sample_rate, 30, 1500, vad)

        # Select the microphone by ID
        mic_id = select_microphone_by_id(mic_id)
        if mic_id is None:
            continue  # Skip if the microphone is not valid
        
        for i, segment in enumerate(segments):
            path = 'chunk-%002d.wav' % (i,)
            print(' Writing %s' % (path,))
            voice_transcriber.write_wave(path, segment, sample_rate)
            result = voice_transcriber.transcribe_speech(path, model)

            print("Transcription:")
            print(result['text'])

            os.remove(path)

def locateMic(name, mic_devices):
    for mic_id, mic_name in mic_devices:
        match = re.search(name, mic_name)
        if match:
            return mic_id


# Main function to start the threads for both microphones
def main():
    # List available microphones
    mic_devices = list_microphones()
    print("Available Microphones:")
    for mic_id, mic_name in mic_devices:
        print(f"ID: {mic_id}, Name: {mic_name}")

    operator_mic = locateMic("NVIDIA Broadcast", mic_devices)
    caller_mic = locateMic("CABLE Output", mic_devices)

    # Starting two threads for two microphones (selecting by ID)
    mic1_thread = threading.Thread(target=handle_microphone_input, args=(mic_devices[operator_mic][0],))  # Using first mic
    mic2_thread = threading.Thread(target=handle_microphone_input, args=(mic_devices[caller_mic][0],))  # Using second mic

    mic1_thread.start()
    mic2_thread.start()
    
    # Join threads to wait for their completion
    mic1_thread.join()
    mic2_thread.join()

if __name__ == "__main__":
    main()
