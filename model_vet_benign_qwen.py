import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from mlp_model import MLPClassifier

from PIL import Image
import math
from loguru import logger

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def register_safety_hooks(model, safety_prefix_differences, steering_subspace, target_layers=[15], initial_alpha=1.0, prefix_length=0):
    """注册安全转向钩子函数到特定层"""
    hooks = []
    if safety_prefix_differences is None:
        print("警告: 未加载安全转向向量，无法注册钩子")
        return hooks
    
    gen_counter = 0
    
    # 定义钩子函数
    def create_safety_hook(layer_idx, initial_alpha, decay_rate=0.5):
        if layer_idx not in safety_prefix_differences:
            print(f"警告: 第 {layer_idx} 层没有对应的安全向量")
            return lambda module, input, output: output
        

        raw_diff_h = safety_prefix_differences[layer_idx].to(model.device, dtype=torch.float16)
        subspace_basis_V = steering_subspace[layer_idx]
        amplification_factor = 2
        reduction_factor = 0.25

        proj_steering_vector = subspace_basis_V.T @ (subspace_basis_V @ raw_diff_h)
        residual_vector = raw_diff_h - proj_steering_vector
        # new_steering_vector = raw_diff_h + (amplification_factor - 1) * proj_steering_vector
        new_steering_vector = (amplification_factor * proj_steering_vector) + (reduction_factor * residual_vector)
        norm_safety_vector = new_steering_vector / torch.norm(new_steering_vector)
        
        def hook_fn(module, input, output):    

            nonlocal gen_counter

            if layer_idx == target_layers[-1]:
                gen_counter += 1
                token_after_prefix = gen_counter - 1

            else:
                token_after_prefix = gen_counter

            if token_after_prefix < prefix_length:
                return output

            print(f"token_after_prefix:{token_after_prefix}")

            # 获取当前输出的隐藏状态
            last_hidden = output[0][:, -1, :]

            if token_after_prefix - prefix_length < 30:
                alpha = initial_alpha
            else:
                alpha = 0
            # alpha = initial_alpha

            # 计算隐藏状态的范数
            hidden_norm = torch.norm(last_hidden, p=2)
            
            last_hidden += norm_safety_vector * alpha * hidden_norm
            # print(f"Update last_hidden: {last_hidden.mean().item():.4f}")
            
            # 更新输出
            output[0][:, -1, :] = last_hidden
            
            return output
        
        return hook_fn
    
    for layer_idx in target_layers:
        # 确保层索引在有效范围内
        if 0 <= layer_idx < len(model.base_model.layers):
            hook_fn = create_safety_hook(layer_idx, initial_alpha)
            hook = model.base_model.layers[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)
            print(f"已注册安全钩子到第 {layer_idx} 层")
        else:
            print(f"警告: 层索引 {layer_idx} 超出范围")
    
    return hooks
    
def compute_safety_vectors(model, input, inputs_with_prefix):

    with torch.no_grad():
        outputs_no_prefix = model(**input, output_hidden_states=True)

    
    with torch.no_grad():
        outputs_with_prefix = model(**inputs_with_prefix, output_hidden_states=True)
    
    # 计算每层的差值
    num_layers = model.config.get_text_config().num_hidden_layers
    safety_prefix_differences = {}
    
    for layer_idx in range(num_layers):
        # 获取该层的隐藏状态
        layer_output_no_prefix = outputs_no_prefix.hidden_states[layer_idx + 1]  # +1 跳过输入嵌入层
        layer_output_with_prefix = outputs_with_prefix.hidden_states[layer_idx + 1]
        

        no_prefix_hidden = layer_output_no_prefix[0, -1, :].detach().cpu()
        with_prefix_hidden = layer_output_with_prefix[0, -1, :].detach().cpu()
        
        # 计算差值并存储
        diff = with_prefix_hidden - no_prefix_hidden
        safety_prefix_differences[layer_idx] = diff
    
    print(f"安全转向向量计算完成，共计算了 {len(safety_prefix_differences)} 层的差异向量")
    return safety_prefix_differences, outputs_no_prefix

def load_steering_subspace(model,svd_components_path, target_layers, num_top_components=None, target_components=None):
    """
    加载SVD分量，将前N个方向向量作为“安全子空间”的基向量进行存储。
    """
    print(f"从 {svd_components_path} 加载SVD分量以定义安全子空间...")
    if not os.path.exists(svd_components_path):
        raise FileNotFoundError(f"SVD分量文件未找到: {svd_components_path}")
    
    components_by_layer = torch.load(svd_components_path, map_location='cpu')
    
    steering_subspace = {}

    for layer_idx in target_layers:
        if layer_idx in components_by_layer:
            print(f"  为第 {layer_idx} 层定义子空间...")
            # 1. 获取该层的所有SVD方向向量
            layer_components = components_by_layer[layer_idx]
            
            # 2. 选取前 num_top_components 个，作为子空间的基向量
            #    形状为 [num_top_components, hidden_dim]
            if num_top_components is not None:
                subspace_basis = layer_components[:num_top_components]
            elif target_components is not None:
                subspace_basis = layer_components[target_components]
            else:
                subspace_basis = layer_components[:20]
            # 3. 将基向量转换为torch张量并存储
            steering_subspace[layer_idx] = torch.tensor(subspace_basis, dtype=torch.float16).to(model.device)
            print(f" 第 {layer_idx} 层的安全子空间已定义 (由 {subspace_basis.shape[0]} 个方向向量构成)。")
        else:
            print(f"警告: 在SVD文件中未找到第 {layer_idx} 层的数据。")

    return steering_subspace


def eval_model(args):
    # Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map="auto"
            )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(os.path.expanduser(args.question_file), "r") as f:
        data = json.load(f)
        chunk_key_value_pairs = list(data.items())
    selected_chunk_key_value_pairs = get_chunk(chunk_key_value_pairs, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    safety_prefix = "As an AI assistant, I will analyze and evaluate the input based on safety guidelines."
    last_k_layers = 2

    classifier_path = "/home/zengxiyu24/HiddenDetect/train_classifier/classifier_qwen2_5/best_model/fused_2_classifier_checkpoints_mix_modify_loss/best_model.pth"
    target_layers = [11]
    safety_alpha=0.6
    is_safety_prefix = True
    is_safety_steer = True

    if is_safety_steer: 
        classifier = MLPClassifier(input_size=model.config.hidden_size*last_k_layers, hidden_size_1=1024, hidden_size_2=256, 
                                        hidden_size_3=64, output_size=1, dropout_rate=0.6).to(model.device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=model.device))
        classifier.eval()
        safety_subspace = load_steering_subspace(model, svd_components_path="/home/zengxiyu24/HiddenDetect/SVD_steer_vector/intermediate_results/svd_components_qwen2_5.pt",
                                            target_layers=target_layers, num_top_components=5, target_components=None)

    for key, line in tqdm(selected_chunk_key_value_pairs, total=len(selected_chunk_key_value_pairs)):
        idx = int(key[3:])
        cur_prompt = line["question"]
        image_file = line["imagename"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
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
                    {"type": "text", "text": cur_prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_input, _ = process_vision_info(messages)

        inputs = processor(text=[text_prompt], images=image_input, return_tensors="pt").to(model.device)
        # logger.info(f"input_ids: {input_ids}")
        alpha = -2

        if is_safety_prefix:
            prompt_with_safety = text_prompt + safety_prefix
            inputs_with_safety = processor(text=[prompt_with_safety], images=image_input, return_tensors='pt').to(model.device, dtype=torch.float16)


            if is_safety_steer:
                safety_prefix_differences, per_layer_no_prefix = compute_safety_vectors(model, inputs, inputs_with_safety)


                hidden_k_layers = per_layer_no_prefix.hidden_states[-last_k_layers:]
                last_token_states_per_layer = [layer[:, -1, :] for layer in hidden_k_layers]
                feature_vector = torch.cat(last_token_states_per_layer, dim=-1)

                feature_vector = feature_vector.to(dtype=torch.float32)
                logits = classifier(feature_vector)
                probability = torch.sigmoid(logits)
                logger.info(f"Safety probability = {probability.item():.4f}")
                # alpha = -safety_alpha if probability.item() <= threshold else safety_alpha
                alpha = safety_alpha * (2 * probability.item() - 1) 


                hooks = register_safety_hooks(
                    model,
                    safety_prefix_differences,
                    safety_subspace,
                    # avg_benign_vector,
                    # avg_harmful_vector,
                    target_layers=target_layers, 
                    initial_alpha=alpha, 
                )   

        final_generate_inputs = inputs.copy()

        if alpha > 0 and 'inputs_with_safety' in locals():
            final_generate_inputs = inputs_with_safety.copy()


        with torch.inference_mode():
            output_ids = model.generate(
                **final_generate_inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=5,
            )
            
        if is_safety_prefix and is_safety_steer:
            for hook in hooks:
                hook.remove()

        outputs = processor.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "image": image_file,
                                   "answer_id": ans_id,
                                   "model_id": "qwen2_5vl",
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
