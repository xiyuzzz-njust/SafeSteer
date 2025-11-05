import torch
import re
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


class Qwen2_5_VL:
    def __init__(self, model_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map="auto"
                )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.temperature, self.num_beams = 0.2, 1

    
    def prepare_inputs(self, sample):    
        processed_text = re.sub(r'\n*<image>\n*', '', sample['txt'])

        image = load_image(sample['img'])
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": processed_text},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        # image_inputs = load_image(sample['img']).resize((224, 224))

        inputs = self.processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(self.model.device)
        return inputs


