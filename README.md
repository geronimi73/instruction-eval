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

response for prompt #1021: `Write a 4 paragraph critique of the following sentence in all capital letters, no lowercase letters allowed: "The processor is clocked at 3.3 ghz". Label each paragraph with SECTION X.`

epoch 0.99:

> SECTION I: The first paragraph should introduce the topic and provide some background information about the sentence being critiqued. It could also include a brief explanation of what the sentence means.
>
> SECTION II: This section should analyze the sentence and its structure. It could discuss the use of capitals, the meaning of the terms used (e.g., \"processor\" and \"ghz\"), and any other relevant details.
>
> SECTION III: In this section, you can evaluate the accuracy of the statement made in the sentence. For example, you could check if the processor really does have a clock speed of 3.3 gHz or if there are any other factors that might impact its performance.
>
> SECTION IV: Finally, you can conclude by summarizing your main points and offering any final thoughts on the sentence.

epoch 5:

> X. The Sentence Under Review
>
> "the PROCESOR IS CLOCKED AT 3.3 GHZ\"
>
> YOU CAN LOSE THE MULTIPLIERS! MORE COVING IN FAMOUS PUNCTUATION!
>
> X. Preamble
>
> I am about to write a really detailed and expert analysis of this sentence, but I have some questions first. What is a \"processor\"? How do you make one \"ghz\"? Is that like a \"turbo boost\" for your computer? I think my computer only goes up to 2.5 ghz, so it's not as cool as your computer. Also, are these numbers over 100? If they are, then this is an over-clocked processor, which is bad. It could potentially cause system instability or data corruption. Why would someone need such a high-speed processor? For video editing? Or gaming? Please tell me it's for gaming, because if it's for work, then I don't get why all these companies are investing into making their technology more powerful.
>
> X. The Actual Ridicule
>
> "3.3 GHz\" - This seems really fast, but I don't know how to compare it to anything else, so I can't say for sure. But I'm pretty sure that my 2018 MacBook Pro doesn't go that fast. Some other computers might be able to though! But here's the thing - who actually needs a computer to go that fast? I don't game or edit video, so I don't need a computer that runs faster than sound. If anything, all this speed just ends up wasting energy and creating more e-waste. Nowadays, we should all be trying to conserve resources, not brag about

## Epoch 1 (20 evals)

![wandb_e1_ife](assets/wandb_e1_ife.png)

![wandb_e1_loss](assets/wandb_e1_loss.png)

[wandb](https://wandb.ai/g-ronimo/huggingface/runs/zd9z4yb1?workspace=user-g-ronimo)

log files: `logs/generations_rnd-724_*.json`

