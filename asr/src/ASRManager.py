import whisper

class ASRManager:
    def __init__(self):
        # initialize the model here
        model = whisper.load_model("base")
        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        result = self.model.transcribe('data/audio.wav', fp16 = False)
        return result['text']
