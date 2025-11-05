import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen2VL:
    def __init__(self, model_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map="auto"
                )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()

    
    def construct_conv_prompt(self, sample):  
        image = Image.open(sample['img']).convert("RGB")
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
                    {"type": "text", "text": sample["txt"]},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        return text_prompt, image_inputs
        
    
    def compute_steering_vector_matrix(self, dataset, safety_prefix, device='cuda'):
        steering_vectors_by_layer = defaultdict(list)

        for sample in tqdm(dataset, desc="计算转向向量"):
            prompt, image_inputs = self.construct_conv_prompt(sample)
            # img = Image.open(sample['img']).convert("RGB")
            # img = img.resize((224, 224)) 
            input = self.processor(text=[prompt], images=image_inputs, return_tensors="pt").to(self.model.device)
            input_with_safety = self.processor(text=[prompt+safety_prefix], images=image_inputs, return_tensors="pt").to(self.model.device)
            # input["attention_mask"] = None

            with torch.no_grad():
                outputs_no_prefix = self.model(**input, output_hidden_states=True)
            # safety_tokens = self.processor.tokenizer(safety_prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
            # # print(f"safety_tokens:{safety_tokens.tolist()}")
            # # decoded_text = self.processor.tokenizer.decode(safety_tokens[0], skip_special_tokens=True)
            # # print(f"解码回来的文本: '{decoded_text}'")
            # input_ids_with_prefix = torch.cat([input["input_ids"], safety_tokens], dim=-1)
            # input["input_ids"] = input_ids_with_prefix
            with torch.no_grad():
                outputs_with_prefix = self.model(**input_with_safety, output_hidden_states=True)
            num_layers = self.model.config.get_text_config().num_hidden_layers
            for layer_idx in range(num_layers):
                state_no_prefix = outputs_no_prefix.hidden_states[layer_idx + 1][:, -1, :].detach().cpu()
                state_with_prefix = outputs_with_prefix.hidden_states[layer_idx + 1][:, -1, :].detach().cpu()
                diff_vector = state_with_prefix - state_no_prefix
                steering_vectors_by_layer[layer_idx].append(diff_vector.squeeze(0))
        steering_matrices = {layer_idx: torch.stack(vectors).numpy() for layer_idx, vectors in steering_vectors_by_layer.items()}
        return steering_matrices
    