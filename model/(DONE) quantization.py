#Code origin
#Author: Alexander Valentini

from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datasets import load_dataset,DatasetDict,Dataset
import time
import glob
import os
import huggingface_hub
import torch

#CALIBRATES THE MODEL TO THE DATA TO QUANTIZE IT

now_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
quant_path = f"checkpoints/quantization/quantize_model_{now_time}"
# put your hugging face token if you want to push the model to hugging face
access_token=None
# change repo id to the place you want to push your model
repoid='AlexVal/name_of_model'
train_save_path="datasets/mcqa/mcqa_train_dataset.jsonl"
# the model repo id need to be quantized
model_path = "AlexVal/mcqa_model_full_only_mcqa"

mcqa_train_data=load_dataset("json", data_files=train_save_path)['train']

# Specify paths and hyperparameters for quantization
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" } # or version: GEMV

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)

data = []
for content in mcqa_train_data:
    msg = content['message']
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())

model.quantize(tokenizer, quant_config=quant_config, calib_data=data)


model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)

#check quantized model size
quantized = False

quantized_checkpoints = glob.glob(f"{quant_path}/*.bin")
if len(quantized_checkpoints) == 0:
    quantized_checkpoints = glob.glob(f"{quant_path}/*.safetensors")
if len(quantized_checkpoints) == 0:
    quantized_checkpoints = glob.glob(f"{quant_path}/*.pt")

quantized_model_size_on_disk = 0
for name in quantized_checkpoints:
    quantized_model_size_on_disk += os.path.getsize(name)
quantized_model_size_gb=quantized_model_size_on_disk/(1024)**3
print("quantization model size (gb):",quantized_model_size_gb)

if access_token is None:
    print("Please set access_token")
    exit(1)

if access_token is not None:
    tokenizer = AutoTokenizer.from_pretrained(quant_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(quant_path,use_safetensors = True).to(device)


    model.push_to_hub(
    repo_id=repoid,
    commit_message="Commiting model",
    token=access_token
    )

    tokenizer.push_to_hub(
    repo_id=repoid,
    commit_message="Commiting tokenizer",
    token=access_token
    )
else:
    print("The model cannot be pushed to HuggingFace since no access token has been provided")
