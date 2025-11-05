import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append("/home/zengxiyu24/HiddenDetect")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from mlp_model import MLPClassifier

from PIL import Image
import math
import time
from loguru import logger

logger.add("logs/benign.log", rotation="1 Mb", level="DEBUG")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["imagename"]
        qs = line["question"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)
        images_size = [img.size for img in [image]]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, images_size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def register_safety_hooks(model, safety_prefix_differences, steering_subspace, target_layers=[15, 16, 17, 18, 19], initial_alpha=1.0, prefix_length=0):
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
        reduction_factor = 0.5

        proj_steering_vector = subspace_basis_V.T @ (subspace_basis_V @ raw_diff_h)
        residual_vector = raw_diff_h - proj_steering_vector

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

            # print(f"token_after_prefix:{token_after_prefix}")

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
        if 0 <= layer_idx < len(model.model.layers):
            hook_fn = create_safety_hook(layer_idx, initial_alpha)
            hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)
            print(f"已注册安全钩子到第 {layer_idx} 层")
        else:
            print(f"警告: 层索引 {layer_idx} 超出范围")
    
    return hooks
    
def compute_safety_vectors(model, tokenizer, safety_prefix, input_ids, image_tensor=None, image_sizes=None):
    
    safety_tokens = tokenizer(safety_prefix, return_tensors='pt').input_ids.cuda()
    safety_tokens = safety_tokens[:, 1:] if safety_tokens[0, 0] == tokenizer.bos_token_id else safety_tokens

    # 带安全前缀的输入
    input_ids_with_prefix = torch.cat([input_ids, safety_tokens], dim=-1)
    
    with torch.no_grad():
        outputs_no_prefix = model(
            input_ids, 
            images=image_tensor,
            image_sizes=image_sizes,
            output_hidden_states=True
        )
    
    with torch.no_grad():
        outputs_with_prefix = model(
            input_ids_with_prefix, 
            images=image_tensor,
            image_sizes=image_sizes,
            output_hidden_states=True
        )
    
    # 计算每层的差值
    num_layers = model.config.num_hidden_layers
    safety_prefix_differences = {}
    
    for layer_idx in range(num_layers):
        # 获取该层的隐藏状态
        layer_output_no_prefix = outputs_no_prefix.hidden_states[layer_idx + 1] 
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

            # subspace_basis = layer_components
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


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file), "r") as f:
        data = json.load(f)
        chunk_key_value_pairs = list(data.items())
    questions_values  = [value for key, value in chunk_key_value_pairs]
    selected_chunk_key_value_pairs = get_chunk(chunk_key_value_pairs, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions_values , args.image_folder, tokenizer, image_processor, model.config)

    safety_prefix = "As an AI assistant, I will analyze and evaluate the input based on safety guidelines."
    last_k_layers = 4
    threshold = 0.5
    classifier_path = "/home/zengxiyu24/HiddenDetect/train_classifier/classifier/best_model/newer/fused_4_classifier_checkpoints_mix_modify_loss/best_model.pth"
    target_layers = [14]
    safety_alpha = 1.0
    is_safety_prefix = True
    is_safety_steer = True

    if is_safety_steer: 
        classifier = MLPClassifier(input_size=model.config.hidden_size*last_k_layers, hidden_size_1=1024, hidden_size_2=256, 
                                        hidden_size_3=64, output_size=1, dropout_rate=0.6).to(model.device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=model.device))
        classifier.eval()
        safety_subspace = load_steering_subspace(model, svd_components_path="/home/zengxiyu24/HiddenDetect/SVD_steer_vector/intermediate_results/svd_components_llava.pt", 
                                            target_layers=target_layers, num_top_components=5, target_components=None)

    for (input_ids, image_tensor, image_sizes), (key, line) in tqdm(zip(data_loader, selected_chunk_key_value_pairs), total=len(selected_chunk_key_value_pairs)):
        idx = int(key[3:])
        cur_prompt = line["question"]
        image_file = line["imagename"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        # logger.info(f"input_ids: {input_ids}")
        alpha = -2

        if is_safety_prefix:
            safety_tokens = tokenizer(safety_prefix, return_tensors='pt').input_ids.cuda()
            forced_decoder_ids = [[i+1, token_id] for i, token_id in enumerate(safety_tokens[0, 1:] if safety_tokens[0, 0] == tokenizer.bos_token_id else safety_tokens[0])]

            if is_safety_steer:
                safety_prefix_differences, per_layer_no_prefix = compute_safety_vectors(model, tokenizer, safety_prefix, input_ids, image_tensor=image_tensor, image_sizes=image_sizes)

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
                    prefix_length=len(forced_decoder_ids) if alpha > 0 else 0,
                )   

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                forced_decoder_ids=forced_decoder_ids if alpha > 0 else None
            )
            
        if is_safety_prefix and is_safety_steer:
            for hook in hooks:
                hook.remove()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "image": image_file,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
