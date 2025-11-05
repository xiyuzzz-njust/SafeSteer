import torch
import sys, os
sys.path.append(os.path.abspath(os.curdir))  
import re
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def load_image_from_bytes(image_data):    
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_images_from_bytes(image_data_list):
    return [load_image_from_bytes(data) for data in image_data_list]

def find_conv_mode(model_name):
    # select conversation mode based on the model name
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"  
    return conv_mode   

class LLaVA:
    def __init__(self, model_path):
        model_name = get_model_name_from_path(model_path)  
        kwargs = {
            "device_map": "auto",    
            "torch_dtype": torch.float16,
            "device": "cuda:0",
        }
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            **kwargs
        )  
        # self.load_scoring_basis_vectors()
        self.conv = find_conv_mode(model_name)
        self.temperature, self.num_beams = 0.2, 1
        self.model.eval()

    def adjust_query_for_images(self, qs):   
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        return qs
    
    def construct_conv_prompt(self, sample):        
        conv = conv_templates[self.conv].copy()  
        if sample['img'] != None:     
            if DEFAULT_IMAGE_TOKEN not in sample['txt']:
                qs = self.adjust_query_for_images(sample['txt'])
            else:
                qs = sample['txt']
        else:
            qs = sample['txt']
        conv.append_message(conv.roles[0], qs)  
        conv.append_message(conv.roles[1], None)       
        prompt = conv.get_prompt()
        return prompt
    
    def prepare_imgs_tensor_both_cases(self, sample):       
        try:
            # Case 1: Comma-separated file paths
            if isinstance(sample['img'], str):
                image_files_path = sample['img'].split(",")
                img_prompt = load_images(image_files_path)
            # Case 2: Single binary image
            elif isinstance(sample['img'], bytes):
                img_prompt = [load_image_from_bytes(sample['img'])]
            # Case 3: List of binary images
            elif isinstance(sample['img'], list):
                # Check if all elements in the list are bytes
                if all(isinstance(item, bytes) for item in sample['img']):
                    img_prompt = load_images_from_bytes(sample['img'])
                else:
                    raise ValueError("List contains non-bytes data.")

            else:
                raise ValueError("Unsupported data type in sample['img']. "
                                "Expected str, bytes, or list of bytes.")
            # Compute sizes
            images_size = [img.size for img in img_prompt if img is not None]
            # Process images into tensor
            images_tensor = process_images(img_prompt, self.image_processor, self.model.config)
            images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)
            return images_tensor, images_size

        except Exception as e:
            print(f"Error preparing image tensors: {e}")
            return None, None
    

    def tokenizer_image_token(self, prompt):
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        ) 
        return input_ids
    

