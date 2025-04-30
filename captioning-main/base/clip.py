from transformers import CLIPProcessor, CLIPModel
from PIL import Image

import time

class CLIP():  
    def __init__(self,config):
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        self.text_model = self.model.text_model
        self.text_projection = self.model.text_projection
        self.logit_scale = self.model.logit_scale
        
    def run(self,image_path, prompt):

        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
        inputs_ = self.processor(text=prompt, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)

        start = time.time()
        out = self.text_model(**inputs_)
        text_embeds = self.text_projection(out[1])
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        print(time.time() - start)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        # end = time.time()
        # print(end-start)
        return probs

