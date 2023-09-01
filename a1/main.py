"""
Some code taken from MIT licensed QLORA repo https://github.com/artidoro/qlora/
"""
from typing import Optional
from ray import tune
import torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, HfArgumentParser, AutoModelForSequenceClassification, AutoTokenizer
import json

@dataclass
class CustomTrainingArguments(TrainingArguments):
    hparam_search: bool = False
    max_length:Optional[int] = None

def model_init(model_url:str, training_args: CustomTrainingArguments):
    model = AutoModelForSequenceClassification.from_pretrained(
            model_url,
            torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
            num_labels = 2, # 2 label in imdb
            )
    return model


def main():
    hfparser = HfArgumentParser((CustomTrainingArguments))

    (training_args,) = hfparser.parse_args_into_dataclasses()
    training_args: CustomTrainingArguments

    model_url ="microsoft/deberta-v3-large"  if training_args.resume_from_checkpoint is None else training_args.resume_from_checkpoint
    
    model = model_init(model_url, training_args)

    tokenizer = AutoTokenizer.from_pretrained(model_url)

    def preprocess_fn(x):
        return tokenizer(x['text'], truncation=True, max_length=training_args.max_length)

    dataset = load_dataset("imdb")

    dataset = dataset.map(preprocess_fn, batched=True)

    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    train_ds_shard = train_ds.shard(index=1, num_shards=20)
    eval_ds_shard = eval_ds.shard(index=1, num_shards=20)


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        incorrect = torch.where(torch.from_numpy(predictions!=labels))[0]
        incorrect = incorrect[torch.randperm(incorrect.shape[0])]

        wrong = []
        for i in incorrect[:10]:
            i = i.item()
            wrong.append({"text":str(eval_ds[i]['text']), "label": int(eval_ds[i]["label"]), "predicted":int(predictions[i])})

        with open("incorrect.json", "w") as f:
            json.dump(wrong, f)


        return {"acc":(predictions == labels).mean()}

    # hparam_search
    best_run = {}
    if training_args.hparam_search:

        trainer = Trainer(
                model_init = lambda : model_init(model_url, training_args),
                args=training_args,
                train_dataset = train_ds_shard,
                eval_dataset = eval_ds_shard,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                )

        def ray_hp_space(_):
            return {
                "learning_rate": tune.loguniform(1e-6, 3e-4),
                "per_device_train_batch_size": tune.choice([training_args.per_device_train_batch_size]),
                "per_device_eval_batch_size": tune.choice([training_args.per_device_eval_batch_size]),
                "iter": 1,

            }

        best_run = trainer.hyperparameter_search(
                hp_space = ray_hp_space,
                n_trials=10, direction="maximize", )

        print(best_run)

    trainer = Trainer(model,
                      training_args,
                      train_dataset = train_ds,
                      eval_dataset = eval_ds,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      **best_run,
                      )

    if training_args.do_train:
        print(trainer.train())

    if training_args.do_eval:
        print(trainer.evaluate())

    trainer.model.save_pretrained("deberta-v3-large-imdb")

if __name__ == "__main__":
    main()


