Use Google's [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)  (https://arxiv.org/abs/2311.07911) to evaluate llama2/mistral *while* training

![wb_small](assets/wb_small.png)

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

Fine-tuning parameters: `mistralai/Mistral-7B-v0.1`, dataset `OpenAssistant/oasst_top1_2023-08-25`, QLoRA: learning rate 0.0001, batch size 8, rank 64, lora_alpha 16.

## Epochs 1-5 (3 evals per epoch)

![W&B Chart 21_11_2023, 07_55_44](assets/W&B%20Chart%2021_11_2023,%2007_55_44.png)

![W&B Chart 21_11_2023, 07_55_34](assets/W&B%20Chart%2021_11_2023,%2007_55_34.png)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/3kgsrfu7?workspace=user-g-ronimo)

log files: `logs/generations_rnd-74_*.json`

### Example 

response for prompt #1051: `Write a letter to a friend in all lowercase letters as to why they should vote for the republican party`

epoch 0.99:

> hi there, i hope you are doing well! i wanted to write and let you know about something important that is coming up soon: the next election. i think it's really important that we all vote, and i want to make sure that you have all the information you need to make an informed decision.
>
> i know that the republican party has been getting a lot of bad press recently, but i think it's important to look at the big picture. the republicans have been working hard to make sure that our country stays strong and prosperous, and i think that their policies will continue to benefit us all if they are re-elected.
>
> i know that some people are worried about the environment, but i think the republicans have a good plan to address those concerns while still keeping our economy growing. and when it comes to social issues, i think the republicans have a better understanding of what is best for our society than the other parties do.
>
> so i just wanted to let you know that i think the republicans are the best choice for our country, and i hope you will consider voting for them in the next election. thanks for listening, and i hope to hear from you soon!

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

![W&B Chart 21_11_2023, 07_59_45](assets/W&B%20Chart%2021_11_2023,%2007_59_45.jpg)

![W&B Chart 21_11_2023, 07_59_36](assets/W&B%20Chart%2021_11_2023,%2007_59_36.jpg)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/x0q4222s?workspace=user-g-ronimo)

log files: `logs/generations_rnd-568_*.json`

