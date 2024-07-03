from typing import List
from PIL import Image
import torch
import io

from transformers import OwlViTProcessor, OwlViTForObjectDetection

class VLMManager:
    def __init__(self):
        # initialize the model here
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        pass

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        img = Image.open(io.BytesIO(image))
        texts = [["a photo of" + caption]]
        inputs = self.processor(text=texts, images=img, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([img.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
        if results[0]["boxes"].size()[0] != 0:
            box = [round(i) for i in results[0]["boxes"][0].tolist()]
        else:
            box = [0,0,1520,870]
        left = box[0]
        top = box[1]
        width = box[2] - left
        height = box[3] - top
        
        return [left, top, width, height]