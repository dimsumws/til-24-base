import whisper

class ASRManager:
    def __init__(self):
        # initialize the model here
        self.model = whisper.load_model("base")
        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # generate audio file
        audiopath = "temp.wav"
        inputAudio = open(audiopath, "wb")
        inputAudio.write(audio_bytes)

        # noise reduction
        audio, rate = librosa.load(audiopath, sr=None)
        noisy_part = audio[0:int(rate*0.3)]  # Identify the noisy part
        reduced_noise_audio = nr.reduce_noise(y=audio, sr=rate, y_noise=noisy_part)

        # perform ASR transcription
        result = self.model.transcribe(reduced_noise_audio)
        return result['text']