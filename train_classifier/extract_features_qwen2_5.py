import torch
import json
import pandas as pd
import random
import os
from tqdm import tqdm
from safetensors.torch import save_file
import random

# 导入您提供的 LLaVA 封装类
from qwen2_5_wrapper import Qwen2_5_VL
from loguru import logger



# --- 配置区域 ---
CONFIG = {
    "model_path": "/home/zengxiyu24/Qwen2.5-VL-7B-Instruct",
    
    # --- 良性数据配置 ---
    "benign_data_path": "/home/zengxiyu24/LLaVA-CC3M-Pretrain-595K/chat.json",
    "benign_image_dir": "/home/zengxiyu24/LLaVA-CC3M-Pretrain-595K/images", 

    "benign_data_bunny_path": "/home/zengxiyu24/LLaVA/playground/data/Bunny_695k/finetune/bunny_695k.json", 
    "benign_image_bunny_dir": "/home/zengxiyu24/LLaVA/playground/data/Bunny_695k/finetune/images", 
    
    # --- 恶意数据配置 ---
    "malicious_data_path": "/home/zengxiyu24/JailBreakV_28K/JailBreakV_28k/JailBreakV_28K.csv",
    "malicious_image_dir": "/home/zengxiyu24/JailBreakV_28K/JailBreakV_28k",


    # 每个数据集抽取的样本数量
    "num_samples_llava_benign": 2000,
    "num_samples_bunny_benign": 10000,
    "num_samples_malicious": 12000,

    # 输出文件保存目录
    "output_dir": "train_classifier/hidden_states_qwen2_5/fused_2_hidden_state_data_mix/"
}

def extract_and_save_states(label, samples_list, model_wrapper, output_path, k):

    hidden_states_dict = {}
    
    for i, sample_info in enumerate(tqdm(samples_list, desc=f"正在处理 {label} 数据")):

        # 1. 准备样本字典以适配model_wrapper
        sample_for_wrapper = {
            'txt': sample_info['txt'],
            'img': sample_info['img_path']  # 传入图像的完整路径
        }

        inputs = model_wrapper.prepare_inputs(sample_for_wrapper)

        # 5. 执行图文前向传播
        with torch.no_grad():
            outputs = model_wrapper.model(**inputs, output_hidden_states=True)

        # 6. 提取最后一个token的隐藏状态
        last_k_layers = outputs.hidden_states[-k:]
        last_token_states_per_layer = [layer[0, -1, :] for layer in last_k_layers]
        fused_hidden_state = torch.cat(last_token_states_per_layer, dim=0).detach().cpu()
        
        # 7. 存入字典
        hidden_states_dict[f'sample.{i}'] = fused_hidden_state


    # 8. 保存到文件
    if hidden_states_dict:
        print(f"正在将 {len(hidden_states_dict)} 个 {label} 隐藏状态保存到: {output_path}")
        save_file(hidden_states_dict, output_path)
        print("保存成功！")
    else:
        print(f"没有成功处理任何 {label} 样本，未生成输出文件。")


def main():
    """主执行函数"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    random.seed(42)  # 设置随机种子以确保结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(42)

    model_wrapper = Qwen2_5_VL(model_path=CONFIG["model_path"])
    print("模型封装类实例化成功！")

    k = 2  # 最后k层的隐藏状态融合


    # --- 处理良性数据 ---
    print("\n--- 开始处理良性图文数据集---")

    benign_samples_list = []

    # 1. 从 LLaVA-CC3M 数据集加载
    with open(CONFIG["benign_data_path"], 'r') as f:
        benign_data = json.load(f)
    
    sampled_benign_data = random.sample(benign_data, CONFIG["num_samples_llava_benign"])
    
    for item in sampled_benign_data:
        if 'conversations' in item and item['conversations'] and item['conversations'][0]['from'] == 'human':
            prompt_text = item['conversations'][0]['value']
            image_filename = item.get('image')
            if image_filename:
                full_image_path = os.path.join(CONFIG["benign_image_dir"], image_filename)
                benign_samples_list.append({'txt': prompt_text, 'img_path': full_image_path})


    # 2. 从 Bunny_695k 数据集加载
    with open(CONFIG['benign_data_bunny_path'], 'r') as f:
        bunny_data = json.load(f)

    sampled_bunny_data = random.sample(bunny_data, CONFIG["num_samples_bunny_benign"])


    for item in sampled_bunny_data:
        if 'conversations' in item and item['conversations'] and item['conversations'][0]['from'] == 'human':
            prompt_text = item['conversations'][0]['value']
            image_filename = item.get('image')
            if image_filename:
                full_image_path = os.path.join(CONFIG["benign_image_bunny_dir"], image_filename)
            else:
                full_image_path = "/home/zengxiyu24/JailBreakV_28K/JailBreakV_28k/llm_transfer_attack/blank_0.png"
            
            if os.path.exists(full_image_path):
                benign_samples_list.append({'txt': prompt_text, 'img_path': full_image_path})
    

    print(f"成功加载并采样 {len(benign_samples_list)} 条良性图文样本。")
    
    if benign_samples_list:
        benign_output_path = os.path.join(CONFIG["output_dir"], "benign_hidden_states_multimodal.safetensors")
        extract_and_save_states("benign", benign_samples_list, model_wrapper, benign_output_path, k)


    # --- 处理恶意数据 ---
    print("\n--- 开始处理恶意图文数据集 (JailBreakV_28k) ---")

    malicious_df = pd.read_csv(CONFIG["malicious_data_path"])

    sampled_malicious_df = malicious_df.sample(n=CONFIG["num_samples_malicious"])

    malicious_samples_list = []
    for _, row in sampled_malicious_df.iterrows():
        prompt_text = row['jailbreak_query']
        image_filename = row.get('image_path')
        if isinstance(image_filename, str) and image_filename: # 确保路径是有效字符串
            full_image_path = os.path.join(CONFIG["malicious_image_dir"], image_filename)
            malicious_samples_list.append({'txt': prompt_text, 'img_path': full_image_path})


    print(f"成功加载并采样 {len(malicious_samples_list)} 条恶意图文样本。")
    
    if malicious_samples_list:
        malicious_output_path = os.path.join(CONFIG["output_dir"], "malicious_hidden_states_multimodal.safetensors")
        extract_and_save_states("malicious", malicious_samples_list, model_wrapper, malicious_output_path, k)


    print("\n所有任务完成！")


if __name__ == "__main__":
    main()