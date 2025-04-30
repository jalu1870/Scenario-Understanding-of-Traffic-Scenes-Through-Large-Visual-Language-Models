import os

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoModelForCausalLM
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

class blip():
    

    models = {"blip-caption-large":'Salesforce/blip-image-captioning-large',
              "blip-caption-base":'Salesforce/blip-image-captioning-base',
              "blip-itm-coco":'Salesforce/blip-itm-large-coco',
              "blip-capfilt-large":'Salesforce/blip-vqa-capfilt-large',
              "blip2-opt":"Salesforce/blip2-opt-6.7b",
              "blip2-t5-xxl":"Salesforce/blip2-flan-t5-xxl",
              "instructblip-13b-v":"Salesforce/instructblip-vicuna-13b"}
    
    def __init__(self,config):
        from transformers import BlipProcessor, Blip2Processor, InstructBlipForConditionalGeneration, BlipForQuestionAnswering, BlipForConditionalGeneration, BlipForImageTextRetrieval, Blip2ForConditionalGeneration, InstructBlipProcessor
        self.model_path = self.models[config["General"]["model"]]
        if(self.model_path == "Salesforce/blip-image-captioning-large"):
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to("cuda")
        if(self.model_path == "Salesforce/blip-image-captioning-base"):
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to("cuda")
        if(self.model_path == "Salesforce/blip-itm-large-coco"):
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForImageTextRetrieval.from_pretrained(self.model_path).to("cuda")
        if(self.model_path == "Salesforce/blip-vqa-capfilt-large"):
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_path).to("cuda")
        if(self.model_path == "Salesforce/blip2-opt-6.7b"):
            self.processor = Blip2Processor.from_pretrained(self.model_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_path).to("cuda")
        if(self.model_path == "Salesforce/blip2-flan-t5-xxl"):
            self.processor = Blip2Processor.from_pretrained(self.model_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_path).to("cuda")
        if(self.model_path == "Salesforce/instructblip-vicuna-13b"):
            self.processor = InstructBlipProcessor.from_pretrained(self.model_path)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_path).to("cuda")
        #self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to("cuda")


    def run(self,image_path, prompt):
        # chat example
        image = Image.open(image_path).convert('RGB')

        # prompt = "Question: does the brightness of the scene overpower the roads visibility? Answer:"
        
        prompt = "Question: " + prompt + " Answer:"
        inputs = self.processor(image, prompt, return_tensors="pt").to("cuda")

        gen_kwargs = {"max_length": 50, "do_sample": False}

        with torch.no_grad():
            if(self.model_path == "Salesforce/blip-itm-large-coco"):
                itm_scores = self.model(**inputs)[0]
                cosine_score = self.model(**inputs, use_itm_head=False)[0]
            #if(self.model_path == "Salesforce/blip-image-captioning-large" or self.model_path == "Salesforce/blip-vqa-capfilt-large"):
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                #outputs = outputs[:, inputs['input_ids'].shape[1]:]
                t = self.processor.decode(outputs[0],skip_special_tokens=True)#.replace("</s>","")
            return t

class cog():
    models = {"cogagent":'THUDM/cogagent-chat-hf',
              "cogagent-vqa":'THUDM/cogagent-vqa-hf',
              "cogvlm":'THUDM/cogvlm-chat-hf'}
    
    def __init__(self,config):
        self.model_path = self.models[config["General"]["model"]]

        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()


    def run(self,image_path, prompt):
        
        image = Image.open(image_path).convert('RGB')
        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[input_by_model['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to('cuda').to(torch.bfloat16)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "temperature": 0.9,
                      "do_sample": False}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            return response
            print("\nCog:", response)
        # image = Image.open(image_path).convert('RGB')
        # inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        # inputs = {
        #     'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        #     'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        #     'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        #     'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        #     'cross_images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]]
        # }
        # gen_kwargs = {"max_length": 2048, "do_sample": False}

        # with torch.no_grad():
        #     outputs = self.model.generate(**inputs, **gen_kwargs)
        #     outputs = outputs[:, inputs['input_ids'].shape[1]:]
        #     t = self.tokenizer.decode(outputs[0]).replace("</s>","")
        #     return t

class deepseek():
    def __init__(self,config):
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
              "composer-hd-336":"internlm/internlm-xcomposer2-4khd-7b",
              "composer-vl":"internlm/internlm-xcomposer2-vl-7b"}
    def __init__(self,config):
        self.model_name = config["General"]["model"]
        self.model_path = self.models[config["General"]["model"]]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)#.cuda()
        # Set `torch_dtype=torch.floatb16` to load model in bfloat16, otherwise it will be loaded as float32 and might cause OOM Error.
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        self.model = self.model.eval()
    
    def run(self,image_path, prompt):
        with torch.cuda.amp.autocast():
            #first:55
            #second:1
            if(self.model_path == "internlm/internlm-xcomposer2-4khd-7b"):
                if(self.model_name == "composer-hd-336"):
                    response, his = self.model.chat(self.tokenizer, query="<ImageHere>" + prompt, image=image_path, hd_num=1, history=[], do_sample=False, num_beams=1,
                        use_cache=True,
                        # output_scores=True,
                        # return_dict_in_generate=True
                        )
                elif(self.model_name == "composer-hd"):
                    response, his = self.model.chat(self.tokenizer, query="<ImageHere>" + prompt, image=image_path, hd_num=55, history=[], do_sample=False, num_beams=1,
                        use_cache=True,
                        # output_scores=True,
                        # return_dict_in_generate=True
                        )
            else:
                response, his = self.model.chat(self.tokenizer, query="<ImageHere>" + prompt, image=image_path, history=[], do_sample=False, num_beams=1,
                    use_cache=True,
                    # output_scores=True,
                    # return_dict_in_generate=True
                    )
        return response

class LLaVA():  
    models = {"llava-1.5":"liuhaotian/llava-v1.5-13b",
              "llava-1.6_7m":"liuhaotian/llava-v1.6-mistral-7b",
              "llava-1.6_7v":"liuhaotian/llava-v1.6-vicuna-7b",
              "llava-1.6_13":"liuhaotian/llava-v1.6-vicuna-13b",
              "llava-1.6_34":"liuhaotian/llava-v1.6-34b"}
    def __init__(self,config):
        from LLaVA.llava.model.builder import load_pretrained_model
        from LLaVA.llava.mm_utils import get_model_name_from_path
        from LLaVA.llava.eval.run_llava import eval_model

        self.model_path = self.models[config["General"]["model"]]

        self.model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=get_model_name_from_path(self.model_path)
        )
        self.eval_model = eval_model
    def run(self,image_path, prompt):
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        model_args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": self.model_name,
            "query": prompt,
            "conv_mode": conv_mode,
            "image_file": image_path,
            "sep": ";",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 40
        })()
        
        # start = time.time()
        with torch.no_grad():
            result = self.eval_model(model_args,self.tokenizer, self.model, self.image_processor, self.context_len)
            # end = time.time()
            # print(end-start)
            return result
