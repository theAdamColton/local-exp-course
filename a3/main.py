'''
Code source (with some changes):
https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234
https://gist.githubusercontent.com/theDestI/fe9ea0d89386cf00a12e60dd346f2109/raw/15c992f43ddecb0f0f857cea9f61cd22d59393ab/explain.py
'''

import torch
import pandas as pd

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import json
from math import ceil
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350

import argparse 
import os 

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    
    def forward_func(self, inputs: torch.Tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
    def visualize(self, inputs, attributes, outfile_path: str):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        a = pd.Series(attr.cpu().numpy()[0][::-1], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1])

        a = a[::-1]

        batch_size = 64

        batched_a = [a[i * batch_size:i*batch_size + batch_size] for i in range(ceil(len(a) / batch_size))] 

        # hack to concat images

        for i, _a in enumerate(batched_a):
            _a = _a[::-1]
            _a.plot.barh(figsize=(2,20), fontsize=11)
            plt.tight_layout(pad=1.15)
            plt.savefig(outfile_path + "__" + str(i) + ".png")

        images = []
        for i in range(len(batched_a)):
            images.append(Image.open(outfile_path + "__" + str(i) + ".png"))

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        new_im.save(outfile_path + ".png")


                      
    def explain(self, text: str, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
                
        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                  return_convergence_delta = True,
                                  internal_batch_size=1,
                                          )
        # Give a path to save
        self.visualize(inputs, attributes, outfile_path)
    
    def generate_inputs(self, text: str) -> torch.Tensor:
        """
            Convenience method for generation of input ids as list of torch torch.Tensors
        """
        return torch.LongTensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False)).to(self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> torch.Tensor:
        """
            Convenience method for generation of baseline vector as list of torch torch.Tensors
        """        
        return torch.LongTensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id]).to(self.__device).unsqueeze(0)

def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base') 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.analsis_dir, exist_ok=True)

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=device,
                                torch_dtype=torch.bfloat16,
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    with open(args.a1_analysis_file, 'r') as f:
        d = json.load(f)
    idx=0
    for obj in d:
        exp_model.explain(obj['text'], os.path.join(args.output_dir,f'example_{idx}'))
        idx+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analsis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='./incorrect.json', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)
