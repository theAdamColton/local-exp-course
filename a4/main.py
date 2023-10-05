"""
This is a modified version of: https://captum.ai/tutorials/TCAV_NLP

This code makes the following changes:
    * Use Huggingface transformer model instead of a conv net, specifically a
        text encoder trained to classify IMDB review sentiment

    * Use Huggingface datasets

    * For the neutral concept, picks random items from IMDB

    * For the positive concept, uses IMDB reviews that include numberic ratings
    in them. (Something like: "I love this movie, 10/10", or "I hate this
    movie, 0/5")

Thus the purpose of this concept activation test is to test the reliance of the classification model on the text ratings that have these explicit scores in them.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import re

from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str


def has_numeral_rating(x):
    """
    returns True if there is something like 10/10 or 4/5 or 2/4 or 1 / 10 in the text
    """
    # basically some number / some number
    # but has to have a space at one end or another,
    # or must end in a period
    pattern = r"\d+ ?\/ ?\d+\s|\s\d+ ?\/ ?\d+|\d+ ?\/ ?\d+\."

    # something like 10/20/2022
    date_pattern = r"\d+\/\d+\/\d+"

    matches = re.findall(pattern, x)

    date_matches = re.findall(date_pattern, x)

    return len(matches) > 0 and len(date_matches) == 0


def main(model_url = "adams-story/deberta-v3-large-imdb", fp16 = True, bf16 = False, device='cuda'):
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(model_url, torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)),).to(device)

    def preprocess_fn(x):
        return tokenizer(x['text'], truncation=True, padding=True, return_tensors="pt")

    dataset = load_dataset("imdb")
    dataset = dataset.map(preprocess_fn, batched=True)

    ds_train = dataset['train'].with_format('torch')
    ds_test = dataset['test'].with_format('torch')

    # finds examples with numeric ratings
    ds_w_numeral_rating = ds_train.filter(lambda x: has_numeral_rating(x['text']))

    ds_w_o_numeral_rating = ds_train.filter(lambda x: not has_numeral_rating(x['text']))

    # prints some
    for x in ds_w_numeral_rating[:5]['text']:
        print(x)
        print()

    n_concepts = 32

    # downsamples the concepts
    ds_w_numeral_rating = ds_w_numeral_rating.shuffle().select(range(n_concepts))
    ds_w_o_numeral_rating = ds_w_o_numeral_rating.shuffle().select(range(n_concepts))

    concept_w_numeral_rating = Concept(0, "w_numeric_rating", ds_w_numeral_rating)
    concept_w_o_numeral_rating = Concept(1, "w_o_numeric_rating", ds_w_o_numeral_rating)

    all_mlp_layers = [n for n,_ in model.named_modules() if 'dense' in n]

    tcav = TCAV(model, layers=all_mlp_layers)
    
    experimental_sets = [[concept_w_numeral_rating, concept_w_o_numeral_rating],]

    # monkey patches the model's forward method
    model.__forward = model.forward
    def new_forward(x):
        input_ids, attention_mask = x['input_ids'], x['attention_mask']
        input_ids = input_ids.to(model.device).unsqueeze(0)
        attention_mask = attention_mask.to(model.device).unsqueeze(0)
        outputs = model.__forward(input_ids, attention_mask)
        logits = outputs.logits
        return logits

    model.forward = new_forward

    

    w_numeral_interpretations = tcav.interpret(preprocess_fn({"text":"I just watched this movie, 0/10."}), experimental_sets=experimental_sets)

    
    import bpdb
    bpdb.set_trace()


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
