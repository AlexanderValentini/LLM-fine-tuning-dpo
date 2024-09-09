#Code origin
#Author: Alexander Valentini

from datasets import load_dataset,DatasetDict,Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM 
from peft import LoraConfig
import os
import torch
import numpy as np
import wandb

project_dir = os.path.dirname(os.path.abspath(os.getcwd()))
previous_checkpoint_path = None

train_data_path = 'datasets/mcqa/mcqa_train_dataset_full_chattemplate_mcqa.jsonl'
vali_data_path = 'datasets/mcqa/mcqa_validation_dataset_full_chattemplate_mcqa.jsonl'

dataset=load_dataset('json', data_files={"train":train_data_path, "validation":vali_data_path})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model_name='stabilityai/stablelm-zephyr-3b'
safetensor_path = 'sft_stablelm_zephyr_1.6b'
tokenizer = AutoTokenizer.from_pretrained("new_tokenizer")
tokenizer.model_max_length = 512

model = AutoModelForCausalLM.from_pretrained(
    safetensor_path,
    attn_implementation="sdpa",
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
).to(device)

wandb_path = os.path.join(project_dir, 'wandb_key_alex.txt')
with open(wandb_path, "r") as f:
        wandb_key = f.read().strip()
        
wandb.login(key=wandb_key)
wandb.init(
    project="MNLP_sft_mcqa_1.6B",
)


#model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    attn_implementation="sdpa"
#).to(device)

print("Running New Version")
print(tokenizer.pad_token)
print(tokenizer.pad_token_id)

response_template = "<|assistant|>\n" 
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer)

#peft_config = LoraConfig(
#        lora_alpha=64,
#        lora_dropout=0.05,
#        r=128,
#        bias="none",
#        target_modules="all-linear",
#        task_type="CAUSAL_LM",
#)

training_args = TrainingArguments(
    output_dir="mcqa_model_full_1.6B_sft_mcqa", # directory to save and repository id
    num_train_epochs=3,                     
    per_device_train_batch_size=1,          
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            #Alexander: use gradient checkpointing to save memory - from tutorial
    optim="adamw_torch_fused",              # Alexander: use fused adamw optimizer for faster training
    logging_steps=10,                       # Alexander: log every 10 steps for better debugging
    save_strategy="steps",
    save_steps= 600,                  
    eval_strategy="steps",
    eval_steps= 300,            
    save_total_limit=3,                     
    load_best_model_at_end=True,            #Load best model at the end of training
    metric_for_best_model="eval_loss",      # metric to use for best model
    learning_rate=5e-6,                     # Lower learning rate to avoid 
    #fp16=True,                             
    bf16=True,                              
    tf32=True,                              
    warmup_ratio=0.1,                      
    lr_scheduler_type="linear",           # use constant learning rate scheduler
    report_to="wandb",                # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=False,
    data_collator=collator,
#    peft_config=peft_config,
)

# start training, the model will be automatically saved to the output directory
if previous_checkpoint_path is not None:
    trainer.train(resume_from_checkpoint=previous_checkpoint_path)
else:
    trainer.train()

trainer.save_model()
