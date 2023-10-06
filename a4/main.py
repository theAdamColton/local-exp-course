"""
This is a modified version of: https://captum.ai/tutorials/TCAV_NLP

This code makes the following changes:
    * Use Huggingface transformer model instead of a conv net, specifically a
        text encoder trained to classify IMDB review sentiment. TCAV requires
        that the text encoder's forward function should output a single value.
        The output is the classifiers logit for the label 0.

    * Monkeypatches captum's classification code because it has a bug.
    my pull request on captums github page: https://github.com/pytorch/captum/pull/1189

    * Use Huggingface datasets

    * Remove padding token symbol from the given examples and concept strings
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import json
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


os.makedirs("figures", exist_ok=True)

from fixed_train_and_eval import train_and_eval

# monkey patches captum to fix bug
import captum
captum.concept._utils.classifier.DefaultClassifier.train_and_eval = train_and_eval


neutral_concept_strings = [
    "with pale blue <unk> in these peaceful",
    "it flows so long as falls the",
    "and that is why   ",
    "when i peruse the conquered fame of",
    "of inward strife for truth and <unk>",
    "the red sword sealed their <unk> ",
    "and very venus of a <unk> ",
    "who the man    ",
    "and so <unk> then a worthless <unk>",
    "to hide the orb of <unk> every",
]

positive_concept_strings = [
    "so well",
    "    so good      ",
    "      love it   ",
    "      like it   ",
    "    even greater      ",
    "  awesome          ",
    "        totally sick dude       ",
    "      fantastical      ",
    "fantastic          ",
    "grand          ",
    "    teriffic",
    "it was, I dare say, stupendous",
]

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
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=6)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=6)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)



def main(model_url = "adams-story/deberta-v3-large-imdb", fp16 = True, bf16 = False, device='cuda', batch_size = 1):
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(model_url, torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)),).to(device)


    dataset = load_dataset("imdb")

    ds_train = dataset['train']
    ds_test = dataset['test']

    def preprocess_fn(text, max_len=1024):
        out =  tokenizer(text, truncation=True, padding='max_length', return_tensors="pt", max_length=max_len)
        input_ids, attention_mask = out['input_ids'], out['attention_mask']
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        return input_ids, attention_mask

    def collate_fn(x):
        return preprocess_fn(x)

    positive_dataloader = DataLoader(positive_concept_strings, batch_size=batch_size, collate_fn=collate_fn)
    neutral_dataloader = DataLoader(neutral_concept_strings, batch_size=batch_size, collate_fn=collate_fn)

    positive_concept = Concept(0, "positive_concept", positive_dataloader)
    neutral_concept = Concept(1, "neutral_concept", neutral_dataloader)

    # I use the output linear layers in the Roberta which are
    # different from the attention output linear layers
    dense_named_modules = [(n,m,) for n,m in model.named_modules() if 'output.dense' in n and 'attention.output.dense' not in n]

    dense_names = [n for n,_ in dense_named_modules]

    # layers should be in depth order, so we should be able to just take the
    # last few and these are the high level layers in the model.
    n_tcav_modules = 16
    tcav_layers = dense_names[-n_tcav_modules:]
    tcav = TCAV(model, layers=tcav_layers)
    
    experimental_sets = [[positive_concept, neutral_concept],]

    # monkey patches the model's forward method
    # because tcav requires that the model returns a single torch scalar per
    # batch item
    model.__forward = model.forward
    def new_forward(*args, **kwargs):
        outputs = model.__forward(*args, **kwargs)
        logits = outputs.logits

        # has to return a single value per batch
        return logits[:,1]

    model.forward = new_forward

    n_inputs = 2
    positive_inputs = ["It was a fantastic play !", "A terrific film so far !", "We loved that show !"]
    positive_inputs = positive_inputs[:n_inputs]
    neutral_inputs = ["It was not a good movie", "I've never watched something as bad", "It is a terrible movie !"]
    neutral_inputs = neutral_inputs[:n_inputs]

    positive_interpretations = tcav.interpret(preprocess_fn(positive_inputs), experimental_sets=experimental_sets)

    plot_tcav_scores(experimental_sets, positive_interpretations, tcav_layers,)
    plt.savefig("figures/tcav_positive.png")
    plt.close()
    
    negative_interpretations = tcav.interpret(preprocess_fn(neutral_inputs), experimental_sets=experimental_sets)
    plot_tcav_scores(experimental_sets, negative_interpretations, tcav_layers,)
    plt.savefig("figures/tcav_negative.png")
    plt.close()

    with open("./incorrect.json") as f:
        d = json.load(f)

    for i, x in enumerate(d):
        interpretations = tcav.interpret(preprocess_fn([x['text']]), experimental_sets=experimental_sets)
        plot_tcav_scores(experimental_sets, interpretations, tcav_layers)
        plt.savefig(f"figures/tcav_example{i:02}.png")
        plt.close()



if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
