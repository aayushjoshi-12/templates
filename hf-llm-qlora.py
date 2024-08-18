import os
import torch
from datasets import load_dataset
from google.colab import userdata
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# pip install -q transformers bitsandbytes peft datasets evaluate trl

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Get access token from user data
access_token = userdata.get("HF_TOKEN")
# access_token = os.environ.get("HF_TOKEN")
# uncomment if using on local machine

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained model name
model_name = "mistralai/Mistral-7B-V0.3"

# Configuration for BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Configuration for Lora
peft_config = LoraConfig(
    r=16, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    max_grad_norm=0.35,
    warmup_ratio=0.03,
    max_steps=100,
    save_steps=100,
    lr_scheduler_type="constant",
)

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
dataset = load_dataset()

# Function to create prompt for each example
def createPrompt(example):
    bos_token = "<s>"
    system_prompt = "[INST] You are a finance suggestion model and your role is to give finance related suggestions \n"
    input_prompt = f" {example['input']} [/INST]"  # depends on the dataset
    output_prompt = f"{example['output']} </s>"

    return [bos_token + system_prompt + input_prompt + output_prompt]

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    formatting_func=createPrompt,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Convert all normalization layers to float32
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train the model
trainer.train()

# Save the trained model
save_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
save_model.save_pretrained("./results/model")

# Load the trained model for testing
lora_config = LoraConfig.from_pretrained("./results/model")
model = get_peft_model(model, lora_config)

# uncomment for testing while training
# text = input(">>> ")
# text = "Question: {text}"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# inputs = tokenizer(text, return_tensors='pt').to(device)

# # Generate output based on user input
# outputs = model.generate(**inputs, max_new_tokens=512)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))