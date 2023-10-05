"""
This is a modified version of: https://captum.ai/tutorials/TCAV_NLP

This code makes the following changes:
    * Use Huggingface transformer model instead of a conv net, specifically a
        text encoder trained to classify IMDB review sentiment. TCAV requires
        that the text encoder's forward function should output a single value.
        The output is the entropy over the classifiers logits.

    * Monkeypatches captum's classification code because it has a bug.
    my pull request on captums github page: https://github.com/pytorch/captum/pull/1189

    * Use Huggingface datasets

    * For the neutral concept, picks random items from IMDB

    * For the positive concept, uses IMDB reviews that include numberic ratings
    in them. (Something like: "I love this movie, 10/10", or "I hate this
    movie, 0/5")

Thus the purpose of this concept activation test is to test the reliance of the
classification model on the text ratings that have these explicit scores in
them. My hypothesis is that the positive concept will decrease the entropy of
the classifiers logits.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import re
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str



# this stops captum from resuming progress
if os.path.exists("./cav/"):
    shutil.rmtree("./cav/")


from fixed_train_and_eval import train_and_eval

# monkey patches captum to fix bug
import captum
captum.concept._utils.classifier.DefaultClassifier.train_and_eval = train_and_eval


def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))


def plot_tcav_scores(experimental_sets, tcav_scores, layers = ['convs.2'], score_type='sign_count'):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)
        
        layers = tcav_scores[concepts_key].keys()
        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores[score_type][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name, fontsize=8)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=6)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)





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


def main(model_url = "adams-story/deberta-v3-large-imdb", fp16 = True, bf16 = False, device='cuda', batch_size = 1):
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(model_url, torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)),).to(device)


    dataset = load_dataset("imdb")

    ds_train = dataset['train']
    ds_test = dataset['test']

    # finds examples with numeric ratings
    ds_w_numeral_rating = ds_train.filter(lambda x: has_numeral_rating(x['text']))

    ds_w_o_numeral_rating = ds_train.filter(lambda x: not has_numeral_rating(x['text']))

    # prints some
    for x in ds_w_numeral_rating[:5]['text']:
        print(x)
        print()

    n_concepts = 64

    # downsamples the concepts
    ds_w_numeral_rating = ds_w_numeral_rating.shuffle()
    ds_w_numeral_rating_concepts = ds_w_numeral_rating.select(range(n_concepts))
    ds_w_numeral_rating = ds_w_numeral_rating.select(range(n_concepts,len(ds_w_numeral_rating)))

    ds_w_o_numeral_rating = ds_w_o_numeral_rating.shuffle()
    ds_w_o_numeral_rating_concepts = ds_w_o_numeral_rating.select(range(n_concepts))
    ds_w_o_numeral_rating = ds_w_o_numeral_rating.select(range(n_concepts,len(ds_w_o_numeral_rating)))


    def preprocess_fn(x, max_len=2048):
        out =  tokenizer(x['text'], truncation=True, padding='max_length', return_tensors="pt", max_length=max_len)
        input_ids, attention_mask = out['input_ids'], out['attention_mask']
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        return input_ids, attention_mask

    def collate_fn(x):
        x = {"text": [y['text'] for y in x]}
        return preprocess_fn(x)

    w_numeral_rating_dataloader = DataLoader(ds_w_numeral_rating_concepts, batch_size=batch_size, collate_fn=collate_fn)
    w_o_numeral_rating_dataloader = DataLoader(ds_w_o_numeral_rating_concepts, batch_size=batch_size, collate_fn=collate_fn)

    concept_w_numeral_rating = Concept(0, "w_numeric_rating", w_numeral_rating_dataloader)
    concept_w_o_numeral_rating = Concept(1, "w_o_numeric_rating", w_o_numeral_rating_dataloader)

    # We want all of the output linear layers in the Roberta, which are
    # different from the attention output linear layers
    dense_named_modules = [(n,m,) for n,m in model.named_modules() if 'output.dense' in n and 'attention.output.dense' not in n]

    dense_names = [n for n,_ in dense_named_modules]


    # layers should be in depth order, so we should be able to just take the
    # last few and these are the high level layers in the model.
    n_tcav_modules = 12
    tcav_layers = dense_names[-n_tcav_modules:]
    tcav = TCAV(model, layers=tcav_layers)
    
    experimental_sets = [[concept_w_numeral_rating, concept_w_o_numeral_rating],]

    # monkey patches the model's forward method
    # because tcav requires that the model returns a single torch scalar per
    # batch item
    model.__forward = model.forward
    def new_forward(*args, **kwargs):
        outputs = model.__forward(*args, **kwargs)
        logits = outputs.logits

        # has to return a single value
        # so we take the entropy of the outputs
        s = F.softmax(logits, dim=-1)
        ls = torch.log(s)
        entropy = - torch.sum(s * ls, dim=-1)
        return entropy

    model.forward = new_forward

    n_inputs = 2
    examples_w_ratings = ds_w_numeral_rating.select(range(n_inputs))
    examples_w_o_ratings = ds_w_o_numeral_rating.select(range(n_inputs))

    examples_w_ratings = preprocess_fn(examples_w_ratings)
    examples_w_o_ratings = preprocess_fn(examples_w_o_ratings)

    w_numeral_interpretations = tcav.interpret(examples_w_ratings, experimental_sets=experimental_sets)

    plot_tcav_scores(experimental_sets, w_numeral_interpretations, tcav_layers,)
    plt.savefig("tcav_w_numeral.png")
    plt.show()
    plt.close()
    
    w_o_numeral_interpretations = tcav.interpret(examples_w_o_ratings, experimental_sets=experimental_sets)
    plot_tcav_scores(experimental_sets, w_o_numeral_interpretations, tcav_layers,)
    plt.savefig("tcav_w_o_numeral.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
