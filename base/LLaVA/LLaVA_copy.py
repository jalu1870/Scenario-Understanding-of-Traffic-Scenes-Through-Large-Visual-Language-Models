import os

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



import torch
from transformers import AutoModelForCausalLM

class deepseek():
    def __init__(self,model_name):
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        from deepseek_vl.utils.io import load_pil_images
        model_path = "deepseek-ai/deepseek-vl-7b-chat"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = vl_gpt.to(torch.bfloat16).cuda().eval()
        self.load_pil_images = load_pil_images
    
    def run(self,image_path, prompt):
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>" + prompt,
                "images": [image_path]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        pil_images = self.load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

class internlm():
    models = {"composer-hd":"internlm/internlm-xcomposer2-4khd-7b",
              "composer-vl":"internlm/internlm-xcomposer2-vl-7b"}
    def __init__(self,model_name):
        self.model_path = self.models[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)#.cuda()
        # Set `torch_dtype=torch.floatb16` to load model in bfloat16, otherwise it will be loaded as float32 and might cause OOM Error.
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        self.model = self.model.eval()
    
    def run(self,image_path, prompt):
        with torch.cuda.amp.autocast():
            response, his = self.model.chat(self.tokenizer, query=prompt, image=image_path, hd_num=55, history=[], do_sample=False, num_beams=1)
            return response

class LLaVA():  
    models = {"llava-1.5":"liuhaotian/llava-v1.5-13b",
              "llava-1.6_13":"liuhaotian/llava-v1.6-vicuna-13b",
              "llava-1.6_34":"liuhaotian/llava-v1.6-34b"}
    def __init__(self,model_name):
        from LLaVA.llava.model.builder import load_pretrained_model
        from LLaVA.llava.mm_utils import get_model_name_from_path
        from LLaVA.llava.eval.run_llava import eval_model

        self.model_path = self.models[model_name]

        self.model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=get_model_name_from_path(self.model_path)
        )
        self.eval_model = eval_model
    def run(self,image_path, prompt):

        model_args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": self.model_name,
            "query": prompt,
            "conv_mode": "llava_v1",
            "image_file": image_path,
            "sep": ";",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        
        # start = time.time()
        result = self.eval_model(model_args,self.tokenizer, self.model, self.image_processor, self.context_len)
        # end = time.time()
        # print(end-start)
        return result
