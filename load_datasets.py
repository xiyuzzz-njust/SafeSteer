
import torch.nn.functional as N
import pandas as pd

import numpy as np
from PIL import Image
from io import BytesIO
import os
import json

import random  
import json
import os

def load_benign_data(file_path):
    safe_set = []
    print(f"loading safe content from JSON{file_path}")
    
    # 加载JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 处理所有条目并格式化为所需的输出格式
    for item in data:
        sample = {
            "txt": item["prompt"],
            "img": item["image_data"],
            "toxicity": 0
        }
        safe_set.append(sample)
            
    return safe_set

def load_safety_eval_data(model_name, file_path):
    unsafe_set = []
    print(f"loading unsafe content from JSON{file_path}")
    # 你需要修改这里的文件路径指向你的实际JSON文件位置
    if model_name == "llava":
        base_path = "data_test/llava_data_test"
    elif model_name == "vlguard":
        base_path = "data_test/vlguard_data_test"
    elif model_name == "llama2":
        base_path = "data_test/minigpt_llama_data_test"
    elif model_name == "qwen2_5":
        base_path = "data_test/qwen2_5_data_test"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    file_path = os.path.join(base_path, file_path)
    

    # 加载JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 处理所有条目并格式化为所需的输出格式
    for item in data:
        sample = {
            "txt": item["prompt"],
            "img": item["image_data"],
            "toxicity": 1  
        }
        unsafe_set.append(sample)
            
    return unsafe_set[:40]


def load_mm_safety_bench(scenario : list):
    dataset = {}
    base_path = "/home/zengxiyu24/MM-SafetyBench/data/processed_questions"
    image_files_base_path = "/home/zengxiyu24/MM-SafetyBench/data/images"
    types = ["SD_TYPO", "SD", "TYPO"]
    
    for s in scenario:
        file_path = os.path.join(base_path, f"{s}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)

        all_items = list(data.items())

        num_to_sample = min(30, len(all_items))
        sampled_items = random.sample(all_items, k=num_to_sample)

        for t in types:
            dataset_name = f"{s}_{t}"
            dataset[dataset_name] = []
            for key, item in sampled_items:
                if t == "SD":
                    qs = item['Rephrased Question(SD)']
                else:
                    qs = item['Rephrased Question']
                    
                image_path = os.path.join(image_files_base_path, s, t, f"{key}.jpg")
                dataset[dataset_name].append({"txt": qs, "img": image_path, "toxicity": 1})

    return dataset