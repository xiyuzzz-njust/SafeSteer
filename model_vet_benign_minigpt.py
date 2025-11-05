import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append("/home/zengxiyu24/HiddenDetect")
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from mlp_model import MiniGPTMLPClassifier

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt_utils.prompt_wrapper import Prompt, minigpt4_vicuna0_prompt, minigpt4_llama2_prompt
from minigpt_utils.generator import Generator

from PIL import Image
import math
from loguru import logger

logger.add("logs/benign.log", rotation="1 Mb", level="DEBUG")

conv_dict = {'pretrain_vicuna0': minigpt4_vicuna0_prompt,
             'pretrain_llama2': minigpt4_llama2_prompt}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



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
            print(f"Update last_hidden: {last_hidden.mean().item():.4f}")
            
            # 更新输出
            output[0][:, -1, :] = last_hidden
            
            return output
        
        return hook_fn
    
    for layer_idx in target_layers:
        # 确保层索引在有效范围内
        if 0 <= layer_idx < len(model.llama_model.base_model.layers):
            hook_fn = create_safety_hook(layer_idx, initial_alpha)
            hook = model.llama_model.base_model.layers[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)
            print(f"已注册安全钩子到第 {layer_idx} 层")
        else:
            print(f"警告: 层索引 {layer_idx} 超出范围")
    
    return hooks
    
def compute_safety_vectors(model, vis_processor, safety_prefix, prompt_text, prompt_defense_text, image=None):
    img = [vis_processor(image).unsqueeze(0).to('cuda')]
    prompt_wrap = Prompt(model=model, 
                        text_prompts=[prompt_text, prompt_text+safety_prefix, prompt_defense_text],
                        img_prompts=[img, img, img],)

    with torch.no_grad():
        outputs_no_prefix = model.llama_model(inputs_embeds=prompt_wrap.context_embs[0], output_hidden_states=True)
        outputs_with_prefix = model.llama_model(inputs_embeds=prompt_wrap.context_embs[1], output_hidden_states=True)
        output_defense = model.llama_model(inputs_embeds=prompt_wrap.context_embs[2], output_hidden_states=True)

    # 计算每层的差值
    num_layers = model.llama_model.config.num_hidden_layers
    safety_prefix_differences = {}
    
    for layer_idx in range(num_layers):
        # 获取该层的隐藏状态
        layer_output_no_prefix = outputs_no_prefix.hidden_states[layer_idx + 1]  # +1 跳过输入嵌入层
        layer_output_with_prefix = outputs_with_prefix.hidden_states[layer_idx + 1]
        
        # 对隐藏状态进行归一化
        no_prefix_hidden = layer_output_no_prefix[0, -1, :].detach().cpu()
        
        with_prefix_hidden = layer_output_with_prefix[0, -1, :].detach().cpu()
        
        # 计算差值并存储
        diff = with_prefix_hidden - no_prefix_hidden
        safety_prefix_differences[layer_idx] = diff
    
    return safety_prefix_differences, output_defense

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
    cfg = Config(args)
    model_config = cfg.model_cfg
    device = f"cuda:{args.gpu_id}"
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    model_type = model_config.model_type
    model.eval()
    generator = Generator(model=model, max_new_tokens=512, device=device,)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(os.path.expanduser(args.question_file), "r") as f:
        data = json.load(f)
        chunk_key_value_pairs = list(data.items())

    selected_chunk_key_value_pairs = get_chunk(chunk_key_value_pairs, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    safety_prefix = "As an AI assistant, I will analyze and evaluate the input based on safety guidelines."
    last_k_layers = 5
    threshold = 0.5
    classifier_path = "train_classifier/classifier_minigpt_llama/best_model/fused_5_classifier_checkpoints_mix_modify_loss_1.0_0.5/best_model.pth"
    target_layers = [15]
    safety_alpha=1.0
    is_safety_prefix = True
    is_safety_steer = True

    if is_safety_steer: 
        classifier = MiniGPTMLPClassifier(input_size=model.llama_model.config.hidden_size*last_k_layers, hidden_size_1=1024, 
                                          hidden_size_2=256, hidden_size_3=64, output_size=1, dropout_rate=0.6).to(model.device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=model.device))
        classifier.eval()
        safety_subspace = load_steering_subspace(model, svd_components_path="SVD_steer_vector/intermediate_results/svd_components_minigpt_llama.pt", 
                                            target_layers=target_layers, num_top_components=5, target_components=None)

    for key, line in tqdm(selected_chunk_key_value_pairs, total=len(selected_chunk_key_value_pairs)):
        idx = int(key[3:])
        cur_prompt = line["question"]
        image_file = line["imagename"]
        image_path = os.path.join(args.image_folder, image_file)

        image = Image.open(image_path).convert("RGB")
        prompt_text = conv_dict[model_type] % cur_prompt
        alpha = -2

        if is_safety_prefix:
            safety_tokens = model.llama_tokenizer(safety_prefix, return_tensors='pt').input_ids.cuda()
            forced_decoder_ids = [[i, token_id] for i, token_id in enumerate(safety_tokens[0, 1:] if safety_tokens[0, 0] == model.llama_tokenizer.bos_token_id else safety_tokens[0])]

            if is_safety_steer:
                safety_prefix_differences, per_layer_no_prefix = compute_safety_vectors(model, vis_processor, safety_prefix, prompt_text, prompt_text, image)

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

        prompt_obj = Prompt(
            model=model,
            text_prompts=[prompt_text],
            img_prompts=[[vis_processor(image).unsqueeze(0).to("cuda:0")]],
            device=device,
        )
            
        if is_safety_prefix and is_safety_steer:
            for hook in hooks:
                hook.remove()

        is_forced=is_safety_steer and alpha > 0
        outputs, _ = generator.generate(prompt=prompt_obj, forced_decoder_ids=forced_decoder_ids if is_forced else None, is_forced=is_forced)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "image": image_file,
                                   "answer_id": ans_id,
                                   "model_id": "minigtp4",
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str, default="/home/zengxiyu24/BAP/eval_configs/minigpt4_llama_eval.yaml", help='Path to the model')
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="others.",
    )
    args = parser.parse_args()

    eval_model(args)
