from PIL import Image
import torch
import sys, os
sys.path.append(os.path.abspath(os.curdir))  

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt_utils.prompt_wrapper import minigpt4_vicuna0_prompt, minigpt4_llama2_prompt, Prompt

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
        minigpt4_chatbot_prompt = conv_dict[self.model_type]   

        prompt_text = minigpt4_chatbot_prompt % sample["txt"]
        return prompt_text

    
    def prepare_inputs(self, sample):

        prompt_text = self.construct_conv_prompt(sample)
        image = Image.open(sample["img"]).convert("RGB")
        
        # 使用你提供的Prompt类来处理输入
        prompt_obj = Prompt(
            model=self.model,
            text_prompts=[prompt_text],
            img_prompts=[[self.vis_processor(image).unsqueeze(0).to(self.device)]],
            device=self.device
        )
        
        # 获取预计算好的上下文embedding
        inputs_embeds = prompt_obj.context_embs[0]
        

        return inputs_embeds
        