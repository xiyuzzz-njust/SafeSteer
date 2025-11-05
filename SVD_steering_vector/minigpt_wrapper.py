import torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt_utils.prompt_wrapper import Prompt, minigpt4_llama2_prompt, minigpt4_vicuna0_prompt
from minigpt_utils.generator import Generator

conv_dict = {'pretrain_vicuna0': minigpt4_vicuna0_prompt,
             'pretrain_llama2': minigpt4_llama2_prompt}

class MiniGPT4:
    def __init__(self, args):
        cfg = Config(args)
        model_config = cfg.model_cfg
        self.device = f"cuda:{args.gpu_id}"
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to('cuda:0')
        self.model_type = model_config.model_type
        self.model.eval()

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    def construct_conv_prompt(self, sample):        
        prompt_text = conv_dict[self.model_type] % sample["txt"]
        return prompt_text

    def compute_steering_vector_matrix(self, dataset, safety_prefix, device='cuda'):
        steering_vectors_by_layer = defaultdict(list)

        for sample in tqdm(dataset, desc="计算转向向量"):
            prompt = self.construct_conv_prompt(sample)
            image = Image.open(sample['img']).convert("RGB")
            img = [self.vis_processor(image).unsqueeze(0).to('cuda')]

            prompt_wrap = Prompt(model=self.model, 
                                text_prompts=[prompt, prompt+safety_prefix],
                                img_prompts=[img, img])

            with torch.no_grad():
                outputs_no_prefix = self.model.llama_model(inputs_embeds=prompt_wrap.context_embs[0], output_hidden_states=True)
                outputs_with_prefix = self.model.llama_model(inputs_embeds=prompt_wrap.context_embs[1], output_hidden_states=True)

            num_layers = self.model.llama_model.config.num_hidden_layers
            for layer_idx in range(num_layers):
                state_no_prefix = outputs_no_prefix.hidden_states[layer_idx + 1][:, -1, :].detach().cpu()
                state_with_prefix = outputs_with_prefix.hidden_states[layer_idx + 1][:, -1, :].detach().cpu()
                diff_vector = state_with_prefix - state_no_prefix
                steering_vectors_by_layer[layer_idx].append(diff_vector.squeeze(0))
        steering_matrices = {layer_idx: torch.stack(vectors).numpy() for layer_idx, vectors in steering_vectors_by_layer.items()}
        return steering_matrices