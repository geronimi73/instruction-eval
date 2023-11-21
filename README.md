Use Google's [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)  (https://arxiv.org/abs/2311.07911) to evaluate llama2/mistral *while* training

![wb_small](assets/wandb_small.png)

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

Both of these numbers are calculated on the prompt level (one prompt consists of ≥1 instructions) and instruction level. Prompt level evaluation is more strict, each prompt is ticked as correct only if all the contained instructions are met in the LLMs reponse.  The authors report all four numbers in the paper, for simplicity the numbers and code below refer to prompt level accuracy only. 

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

Fine-tuning parameters: `mistralai/Mistral-7B-v0.1`, dataset `OpenAssistant/oasst_top1_2023-08-25`, QLoRA: learning rate 0.0001, batch size 8, rank 64, lora_alpha 16.

## Epochs 1-5 (3 evals per epoch)

![wandb_e1-5_ife](assets/wandb_e1-5_ife.png)

![wandb_e1-5_loss](assets/wandb_e1-5_loss.png)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/hq9jjucm?workspace=user-g-ronimo)

log files: `logs/generations_rnd-391_*.json`

### Example 

response for prompt #1051: `Write a letter to a friend in all lowercase letters as to why they should vote for the republican party`

epoch 0.99:

> hi there, i hope you are doing well! i wanted to talk to you about something important that is coming up soon: the next election. i know it can be hard to decide who to vote for, but i think you should consider voting for the republican party. here's why:
>
> 1. the republican party cares about the people. they want to make sure that everyone has a chance to succeed and live their best life. they believe in helping those who need it most, and they work hard to make sure that happens.
> 2. the republican party wants to keep our country strong and safe. they understand that we need to protect ourselves from threats both inside and outside of our borders. they will do everything they can to keep us secure.
>
> 3. the republican party believes in freedom and liberty. they want to make sure that every american has the opportunity to pursue their dreams without being held back by government bureaucracy or red tape. they believe in individual responsibility and personal accountability.
> 4. finally, the republican party stands for traditional values like family, faith, and community. they understand that these things are what make America great, and they will work to protect them at all costs.
> 
>i know this may not be enough information to make up your mind, but i hope it helps give you some insight into why i think you should vote for the republican party. if you have any questions or concerns, please don't hesitate to ask me!

epoch 5:

> hi friend,
>
> i hope you are doing well. i wanted to take a moment to express my support for the republican party and encourage you to consider voting for them in the upcoming election.
>
> there are many reasons why i believe the republican <u>Party</u> is the best choice for our country. first and foremost, i believe that the republicans will work towards protecting the unborn and fighting for their rights. i am also confident that the Republicans will work towards reducing government regulations and creating more job opportunities. <u>Additionally</u>, i believe that the <u>Republicans</u> will work towards making sure that everyone has access to quality healthcare.\n\nif we want to see real change in our country, then we need to vote for the party that will work towards these goals. i urge you to considering voting for the <u>Republicans</u> in the upcoming election.
>
> <u>Thank</u> you for your time, and i look forward to hearing from you soon.
>
> sincerely,
>
> [your name]

## Epoch 1 (20 evals)

![wandb_e1_ife](assets/wandb_e1_ife.png)

![wandb_e1_loss](assets/wandb_e1_loss.png)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/zd9z4yb1?workspace=user-g-ronimo)

log files: `logs/generations_rnd-724_*.json`

