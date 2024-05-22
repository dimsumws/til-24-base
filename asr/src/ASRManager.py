import whisper

class ASRManager:
    def __init__(self):
        # initialize the model here
        self.model = whisper.load_model("base")
        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        audio = open("temp.wav", "wb")
        audio.write(audio_bytes)
        result = self.model.transcribe("temp.wav")
        return result['text']
        #return "test"
