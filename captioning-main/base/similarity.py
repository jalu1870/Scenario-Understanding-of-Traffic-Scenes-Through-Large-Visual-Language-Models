from transformers import pipeline, AutoTokenizer
import numpy as np

class Classifier():
    def __init__(self):
        task = "zero-shot-classification"
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, _from_pipeline=task, max_length=2048
            )
        tokenizer.model_max_length = 2048
        self.classifier = pipeline(task, tokenizer=tokenizer, model=model_name)



    def run(self,output, options):
        output = self.classifier(output, options, multi_label=False)
        labels = output["labels"]
        argm = np.argmax(output["scores"])
        
        return labels[argm]
        print(output)
