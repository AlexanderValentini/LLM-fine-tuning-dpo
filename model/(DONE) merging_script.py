#Code origin
#Author: Alexander Valentini

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
import numpy as np
#Using quantization, but this should probably not be in the final version:
from pathlib import Path
import os

 
#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    llm_int8_threshold=6.0,
#    llm_int8_has_fp16_weight=False,
#    bnb_4bit_compute_dtype=torch.bfloat16,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type="nf4",
#)

#model_name = 'stabilityai/stablelm-zephyr-3b'
#model_name = 'AlexVal/dpo_model'

#safetensors_path = 'sft_stablelm_zephyr_3b/29.05.2024_v4'
#safetensors_path = 'models/dpo_model/merged_model'
safetensors_path = 'dpo_model_1.6B'
#safetensors_path = 'models/sft_stablelm_zephyr_1.6b'
#safetensors_path = 'mcqa_model/merged_model'
#os.path.abspath(os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoPeftModelForCausalLM.from_pretrained(
     safetensors_path,
     torch_dtype=torch.float16,
     attn_implementation="sdpa",

).to(device)

merged_model = model.merge_and_unload()
merged_model.save_pretrained('dpo_model_1.6B/merged_model',safe_serialization=True)

