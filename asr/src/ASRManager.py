from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from pathlib import Path
import librosa

class ASRManager:
    def __init__(self):
        # initialize the model here
        model_path = Path(f"asrModel8")
        proc_path = Path(f"asrModel8proc")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(proc_path)
        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # generate audio file
        audiopath = "temp.wav"
        inputAudio = open(audiopath, "wb")
        inputAudio.write(audio_bytes)
        
        audio_data, sampling_rate = librosa.load(audiopath, sr=None)
        input_features = self.processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]