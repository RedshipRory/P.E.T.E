import caller_transcriber
import whisper
import ollama
import webrtcvad
import os




def main():
    model = whisper.load_model("base")

    while True:
        sample_rate = 24000
        agressiveness = 0
        vad = webrtcvad.Vad(int(agressiveness))
        segments = voice_transcriber.vad_collector(sample_rate, 30, 1500, vad)
        for i, segment in enumerate(segments):
            path = 'chunk-%002d.wav' % (i,)
            print(' Writing %s' % (path,))
            voice_transcriber.write_wave(path, segment, sample_rate)
            result = voice_transcriber.transcribe_speech(path, model)

            print("Transcription:")
            print(result['text'])

            os.remove(path)

if __name__ == "__main__":
    main()