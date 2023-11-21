Use Google's [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)  (https://arxiv.org/abs/2311.07911) to evaluate llama2/mistral *while* finetuning

# IFEval Dataset

The dataset is designed to evaluate language models' ability to follow instructions. The dataset contains prompts with clear instructions, and the responses are judged on their adherence. 

`instruction_following_eval/data/input_data.jsonl`: original data from the Google github repo

examples:

```json
{
	"key": 1001, 
	"instruction_id_list": ["punctuation:no_comma"], 
	"prompt": "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.", 
	"kwargs": [{}]
}
```

```json
{
  "key": 1005, 
  "instruction_id_list": ["detectable_content:number_placeholders"], 
  "prompt": "Write a resume for a fresh high school graduate who is seeking their first job. Make sure to include at least one placeholder represented by square brackets, such as [address].", 
  "kwargs": [{"num_placeholders": 1}]
}
```

for each instruction, two accuracies are calculated:

- `ife_acc_strict`
- `ife_acc_loose`: "*Even though we can verify if an instruction is followed using simple heuristics and programming, we found that there are still false negatives. For example, for a given verifiable instruction of “end your email with: P.S. I do like the cake”, a language model may follow the instruction by ending the email with “P.S. **I do like the cake**” which has markdown tags (** indicates the bold text). If we simply check the string match of “P.S. I do like the cake”, we will miss-classify it as not-followed. To alleviate this false negative problem, we compute a loose accuracy score of instruction following*"

# Code

## test_ife.py

simple test case

```bash
python3 test_ife.py 
{'ife_acc_strict': 0.6666666666666666, 'ife_acc_loose': 0.6666666666666666}
```

## qlora_distributed.py

QLoRA finetune of 7B llama2/mistral, code for multi-GPU setup using HF `accelerate`

Main changes include an evaluation hook:

```python
trainer.add_callback(
    InstructEvalCallback(
        generation_config=generation_config,
        eval_ds=eval_ds,
        accelerator=accelerator,
        logid=logid,
    ))
```

Modify the parameters to fit your setup:

```python
..
modelpath="models/Mistral-7B-v0.1"
lr=0.0001
bs=8        # batch size
ga_steps=1  # gradient acc. steps
epochs=1
evals_per_epoch=20
run_name=modelpath.split("/")[-1]+f"-LR-{lr}_BS-{bs}-{logid}"
use_wandb=True
..
```

```python
..
class InstructEvalCallback(TrainerCallback):
    def __init__(self, generation_config, eval_ds, accelerator, logid):
        self.generation_config = generation_config
        self.eval_ds_all=eval_ds
        self.template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        self.bs=32     # <--- this one, per device batch size for generate()
        self.eval_num=0
        self.accelerator=accelerator
        self.logid=logid
...  

```

run with 

```
accelerate launch qlora_distributed.py
```

## instruction_following_eval/evaluation_main.py

Modifications include:

- A bug fix ([issue](https://github.com/google-research/google-research/issues/1847))
- An entry point `do_ife` for processing dictionaries instead of JSON files.

# Results

Fine-tuning parameters: QLoRA, learning rate 0.0001, batch size 8, rank 64, lora_alpha 16, dataset `OpenAssistant/oasst_top1_2023-08-25`.

## Epochs 1-5

![W&B Chart 21_11_2023, 07_55_44](assets/W&B%20Chart%2021_11_2023,%2007_55_44.png)

![W&B Chart 21_11_2023, 07_55_34](assets/W&B%20Chart%2021_11_2023,%2007_55_34.png)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/3kgsrfu7?workspace=user-g-ronimo)

log files: `logs/generations_rnd-74_*.json`

## Epoch 1 (20 evals)

![W&B Chart 21_11_2023, 07_59_45](assets/W&B%20Chart%2021_11_2023,%2007_59_45.jpg)

![W&B Chart 21_11_2023, 07_59_36](assets/W&B%20Chart%2021_11_2023,%2007_59_36.jpg)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/x0q4222s?workspace=user-g-ronimo)

log files: `logs/generations_rnd-568_*.json`

