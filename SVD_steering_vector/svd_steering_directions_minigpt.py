import argparse
import os
import json
import torch
import numpy as np

from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.curdir))  
from minigpt_wrapper import MiniGPT4

from utils import load_safety_eval_data, load_benign_data

# --- 全局设置与辅助函数 ---
SAFETY_PREFIX = "As an AI assistant, I will analyze and evaluate the input based on safety guidelines."


def perform_svd(steering_matrices):
    print("🔬 正在对矩阵进行SVD分解...")
    components_by_layer, eigenvalues_by_layer = {}, {}
    for layer_idx, matrix in tqdm(steering_matrices.items(), desc="SVD分解"):
        _, s, Vt = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)
        components_by_layer[layer_idx] = Vt
        eigenvalues_by_layer[layer_idx] = s**2 / (matrix.shape[0] - 1)
    print("✅ SVD分解完成！")
    return components_by_layer, eigenvalues_by_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析安全转向向量的内在维度")
    parser.add_argument("--svd-path", type=str, default="/home/zengxiyu24/HiddenDetect/SVD_steer_vector/intermediate_results", help="记录中间结果")
    parser.add_argument("--output-dir", type=str, default="SVD_steer_vector/outputs", help="分析结果输出目录")
    parser.add_argument("--cfg_path", default="/home/zengxiyu24/BAP/eval_configs/minigpt4_llama2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="others.",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    model_wrapper = MiniGPT4(args)
    model_name = "llama2"

    dataset_names = ["original_query.json", "figstep_query.json", "qr_query.json", "shuffle_query.json", "vet_query.json", "bap_query.json", "umk_query.json", "hades_query.json"]
    dataset_for_analysis = []

    for dataset_name in dataset_names:
        dataset_for_analysis.extend(load_safety_eval_data(model_name, dataset_name))

    dataset_for_analysis.extend(load_benign_data("/home/zengxiyu24/HiddenDetect/data/benign_query.json"))

    matrix_path = os.path.join(args.svd_path, "steering_matrices_minigpt_llama.pt")
    components_path = os.path.join(args.svd_path, "svd_components_minigpt_llama.pt")
    if os.path.exists(matrix_path) and os.path.exists(components_path):
        print("检测到已有缓存，直接加载转向矩阵和SVD分解结果...")
        steering_matrices = torch.load(matrix_path)
        components_by_layer = torch.load(components_path)
    else:
        steering_matrices = model_wrapper.compute_steering_vector_matrix(dataset_for_analysis, SAFETY_PREFIX)
        torch.save(steering_matrices, matrix_path)
        components_by_layer, _ = perform_svd(steering_matrices)
        torch.save(components_by_layer, components_path)

    print("\n✅ 分析全部完成！")