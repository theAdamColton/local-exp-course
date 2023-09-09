import os
import torch
import numpy as np
import jsonlines
import random
from transformers import (
    GenerationConfig,
    IdeficsForVisionText2Text,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from datasets import load_dataset

torch.set_grad_enabled(False)

def newyorker_caption_contest_data(task_name, subtask):
    dset = load_dataset(task_name, subtask)

    res = {}
    for spl, spl_name in zip(
        [dset["train"], dset["validation"], dset["test"]], ["train", "val", "test"]
    ):
        cur_spl = []
        for inst in list(spl):
            inp = inst["from_description"]
            targ = inst["label"]
            cur_spl.append(
                {
                    "input": inp,
                    "target": targ,
                    "instance_id": inst["instance_id"],
                    "image": inst["image"],
                    "caption_choices": inst["caption_choices"],
                }
            )

            #'input' is an image annotation we will use for a llama2 e.g. "scene: the living room description: A man and a woman are sitting on a couch. They are surrounded by numerous monkeys. uncanny: Monkeys are found in jungles or zoos, not in houses. entities: Monkey, Amazon_rainforest, Amazon_(company)."
            #'target': a human-written explanation
            #'image': a PIL Image object
            #'caption_choices': is human-written explanation

        res[spl_name] = cur_spl
    return res


def main(
    seed: int = 42,
    task_name: str = "jmhessel/newyorker_caption_contest",
    subtask: str = "explanation",
    mmllm_model_name_or_path: str = "HuggingFaceM4/idefics-9b-instruct",
    llm_model_name_or_path: str = "upstage/llama-30b-instruct-2048",
    quantize_mmllm:bool = False,
    quantize_llm:bool = True,
    beams: int = 2,
):
    os.makedirs("./out", exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    nyc_data = newyorker_caption_contest_data(task_name, subtask)

    """
    ====================================================================================================
        Uses the multimodal model, run over images
    ====================================================================================================
    """
    print("Loading mllm")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
    )

    model = IdeficsForVisionText2Text.from_pretrained(
        mmllm_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=bnb_config if quantize_mmllm else None,
    )

    processor = AutoProcessor.from_pretrained(mmllm_model_name_or_path)

    print("Loading data")
    nyc_data_five_val = random.sample(nyc_data["val"], 5)
    nyc_data_train_two = random.sample(nyc_data["train"], 2)

    prompt = [
        "User: I have some comics from the New Yorker. ",
        "First, write a short description of the comic. ",
        "This includes the scene, anything uncanny or unusual in the comic, along with any entities/characters in the image. Prefix this with 'Description -'\n",
        "Next provide an outline of a joke based on what is going on in the comic. Start this as 'Joke outline -'\n",
        "Then provide the caption, prefixed by 'Hilarious caption -'. The caption should complement the image as an absolutely histericaly funny punchline.\n",
        "Here is the first comic:\n",
    ]
    for i, x in enumerate(nyc_data_train_two):
        prompt += [
            "\nUser: Zinger! Here is the next comic:\n" if i > 0 else "",
            x['image'],
            "<end_of_utterance>"
            "\nAssistant: ",
            "Description - " + x['input'].split("description:")[1].split('caption:')[0][:32],
            "\nJoke outline - ",
            x['target'][:32],
            "\nHilarious caption - ",
            x['caption_choices'],
            "<end_of_utterance>",
        ]

    prompts = [
            [
                *prompt,
                "\nUser: Zinger! Here is the next comic. Remember to write the 'Description', 'Joke outline', and 'Hilarious caption'.\n",
                x['image'],
                "<end_of_utterance>",
                "\nAssistant: Description - ",
            ]
            for x in nyc_data_five_val
        ]
    
    for val_inst in nyc_data_five_val:
        val_inst["image"].save(f"out/{val_inst['instance_id']}-{val_inst['caption_choices']}.jpg")

    # Generation args
    exit_condition = processor.tokenizer(
        "<end_of_utterance>", add_special_tokens=False
    ).input_ids
    bad_words_ids = processor.tokenizer(
        ["<image>", "<fake_token_around_image>"], add_special_tokens=False
    ).input_ids

    force_words_ids = processor.tokenizer(["Hilarious caption -", "Description -", "Joke outline -"], add_special_tokens=False).input_ids

    generation_config = GenerationConfig(
            #force_words_ids=force_words_ids,
            num_beams=beams,
    )

    all_generated_ids = []

    for prompt in prompts:
        inputs = processor([prompt], add_end_of_utterance_token=False, return_tensors="pt").to(device)

        generated_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            repetition_penalty=1.5,
            max_length=1024,
        )

        all_generated_ids.append(generated_ids.cpu())

    generated_text = [processor.decode(generated_ids[0]) for generated_ids in all_generated_ids]

    for i, t in enumerate(generated_text):
        print(f"{i}:\n{t}\n")
        gen_expl = t.split("Assistant:")[-1]
        nyc_data_five_val[i]["generated_idefics"] = gen_expl

    # ======================> You will need to `mkdir out`
    filename = "out/val.jsonl"
    with jsonlines.open(filename, mode="w") as writer:
        for item in nyc_data_five_val:
            del item["image"]
            writer.write(item)

    filename = "out/train.jsonl"
    with jsonlines.open(filename, mode="w") as writer:
        for item in nyc_data_train_two:
            del item["image"]
            writer.write(item)

    model = None
    torch.cuda.empty_cache()


    """
    ====================================================================================================
        Only the LLM, run over text descriptions
    ====================================================================================================
    """

    print("Loading data")
    nyc_data_five_val = []
    with jsonlines.open("out/val.jsonl") as reader:
        for obj in reader:
            nyc_data_five_val.append(obj)

    nyc_data_train_two = []
    with jsonlines.open("out/train.jsonl") as reader:
        for obj in reader:
            nyc_data_train_two.append(obj)

    print("Loading llm")
    """
    Ideally, we'd do something similar to what we have been doing before: 

        tokenizer = AutoTokenizer.from_pretrained(llama2_checkpoint, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(llama2_checkpoint, torch_dtype=torch.float16, device_map="auto")
        tokenizer.pad_token = tokenizer.unk_token_id
        
        prompts = [ "our prompt" for val_inst in nyc_data_five_val]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        output_sequences = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_text = [tokenizer.decode(s, skip_special_tokens=True) for s in output_sequences]

    But I cannot produce text with this prototypical code with HF llama2. 
    Thus we will use pipeline instead. 
    """

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config if quantize_llm else None,
    )

    prompts = [[x for x in prompt if type(x) == str] for prompt in prompts]
    prompts = ["".join(prompt) for prompt in prompts]
    prompts = [prompt.replace("User:", "### User:").replace("Assistant:", "### Assistant:") for prompt in prompts]
    prompts = [prompt + "Description - " + x['input'].split("description:")[1].split('caption:')[0] for prompt, description in zip(prompts, nyc_data_five_val)]
    inputs = tokenizer(prompts, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_length=2048)

    sequences = tokenizer.batch_decode(generated_ids)

    import bpdb
    bpdb.set_trace()
    gen_expl = sequences[0]["generated_text"].split("/INST] ")[-1]
    nyc_data_five_val["generated_llama2"] = gen_expl

    filename = "out/val.jsonl"
    with jsonlines.open(filename, mode="w") as writer:
        for item in nyc_data_five_val:
            writer.write(item)


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
