import torch
import os
import json
import wandb
import numpy as np
from random import randint
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, BitsAndBytesConfig, GenerationConfig, TrainerCallback, TrainerState, TrainerControl
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset, DatasetDict, Dataset
from copy import deepcopy
from tqdm import tqdm
from instruction_following_eval import evaluation_main
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def read_jsonl(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line, strict=False))
        f.close()
        return json_list

accelerator = Accelerator()

modelpath="models/Mistral-7B-v0.1"
eval_ds_file="instruction_following_eval/data/input_data.jsonl"     # https://github.com/google-research/google-research/tree/master/instruction_following_eval
lr=0.0001
bs=8        # batch size
ga_steps=1  # gradient acc. steps
epochs=5
evals_per_epoch=3
logid=randint(0,1000)
run_name=modelpath.split("/")[-1]+f"-LR-{lr}_BS-{bs}-{logid}"
use_wandb=True

set_seed(42)

# Load Instruct-Eval dataset
eval_ds=read_jsonl(eval_ds_file)

if not use_wandb:
    wandb.init(mode="disabled") 

# Load model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map={"": accelerator.process_index},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False, legacy=False)    # fast tokenizer sometimes ignores the added tokens

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token, 
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id

# Add adapters to model
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
    lora_dropout=0.1, 
    bias="none", 
    modules_to_save = ["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.config.use_cache = False

# Load dataset
dataset = load_dataset("OpenAssistant/oasst_top1_2023-08-25")

# Tokenize dataset
def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])

    input_ids,labels,attention_masks = [],[],[]
    for tokens in tokenlist:
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
        attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }
    return batch

steps_per_epoch=len(dataset_tokenized["train"])//(accelerator.state.num_processes*bs*ga_steps)
args = Seq2SeqTrainingArguments(
    output_dir="out",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch//evals_per_epoch,
    save_steps=steps_per_epoch,
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=lr,
    group_by_length=True,
    fp16=True,
    ddp_find_unused_parameters=False,
    run_name=run_name,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
)

generation_config = {
    "temperature": 0.1,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 400,
    "pad_token_id": tokenizer.pad_token_id
}

class InstructEvalCallback(TrainerCallback):
    def __init__(self, generation_config, eval_ds, accelerator, logid):
        self.generation_config = generation_config
        self.eval_ds_all=eval_ds
        self.template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.bs=40      # batch size for generate()
        self.eval_num=0
        self.accelerator=accelerator
        self.logid=logid
    
    def prepare_prompts(self, prompts, tokenizer):
        tokenizer.padding_side="left"     # left pad for inference

        prompts_tok=tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=False, 
            pad_to_multiple_of=8,
            add_special_tokens=False).to("cuda")

        tokenizer.padding_side="right"

        return prompts_tok

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, tokenizer, eval_dataloader, **kwargs):
        self.eval_num+=1   # keep track of how many evals we did already
        model.eval()
        model.config.use_cache = True

        # split questions among GPUs
        with accelerator.split_between_processes(self.eval_ds_all) as eval_ds:
            responses=deepcopy(eval_ds)    # model generation stored here

            # batched inference on each GPU
            batches = [eval_ds[i:i + self.bs] for i in range(0, len(eval_ds), self.bs)]  
            for b_no, batch in enumerate(tqdm(batches, desc=f"GPU {accelerator.process_index}")):
                prompts=[ self.template.format(prompt=item["prompt"]) for item in batch ]   # apply ChatML  
                prompts_tok=self.prepare_prompts(prompts, tokenizer)

                with torch.no_grad():
                    outputs_tok=model.generate(**prompts_tok,**self.generation_config).to("cpu")
                outputs=[
                    tokenizer.decode(
                        outputs_tok[i][outputs_tok[i]!=tokenizer.pad_token_id], 
                        spaces_between_special_tokens=False,
                        skip_special_tokens=False)
                    for i,t in enumerate(outputs_tok) ]

                for i, output_raw in enumerate(outputs):
                    output=output_raw[len(prompts[i]):].strip()     # cut prompt
                    responses[b_no*self.bs+i]["response"]=output
                    responses[b_no*self.bs+i]["GPU"]=accelerator.process_index

                del prompts_tok, outputs_tok
                torch.cuda.empty_cache()

        resp_gathered=gather_object(responses)  # collect results from all GPUs

        # cut at eos if exists in pred.
        eos_str="<|im_end|>"
        for r in resp_gathered:
            if eos_str in r["response"]:
                r["response"]=r["response"].split(eos_str)[0]

        # log to trainer
        ife_result=evaluation_main.do_ife(response_dict=resp_gathered)
        trainer.log(ife_result)

        # log to file
        if accelerator.is_main_process:
            ife_result["num_samples"]=len(resp_gathered)
            ife_result["responses"]=resp_gathered
            write_pretty_json(f"generations_rnd-{logid}_{self.eval_num}.json",ife_result)

        model.config.use_cache = False

        return control

trainer.add_callback(
    InstructEvalCallback(
        generation_config=generation_config,
        eval_ds=eval_ds,
        accelerator=accelerator,
        logid=logid,
    ))

# evaluate after first step for baseline
class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True
# trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()

