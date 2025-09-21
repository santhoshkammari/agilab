import torch
import transformers
import trl
import os
os.environ["WANDB_DISABLED"] = "true"

print(f"📦 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"📊 TRL version: {trl.__version__}")


from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, HTML, Markdown
import torch

model_id = "LiquidAI/LFM2-700M" # <- or LFM2-700M or LFM2-350M

print("📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="auto",
#   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)

print("✅ Local model loaded successfully!")
print(f"🔢 Parameters: {model.num_parameters():,}")
print(f"📖 Vocab size: {len(tokenizer)}")
print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

from datasets import load_dataset

print("📥 Loading SFT dataset...")
train_dataset_sft = load_dataset("Team-ACE/ToolACE",split="train[:80%]")
eval_dataset_sft = load_dataset("Team-ACE/ToolACE", split="train[80%:]")

print("✅ SFT Dataset loaded:")
print(f"   📚 Train samples: {len(train_dataset_sft)}")
print(f"   🧪 Eval samples: {len(eval_dataset_sft)}")
print(f"\n📝 Single Sample: {train_dataset_sft[0]}")

from datasets import load_dataset
import json
import re

def convert_to_lfm2_format(sample):
    """Convert ToolACE format to LFM2 chat template"""

    # Extract system prompt and available functions
    system_prompt = sample['system']
    conversations = sample['conversations']

    # Build the formatted conversation
    formatted_text = "<|startoftext|><|im_start|>system\n"

    # Add system prompt and extract functions if present
    if 'functions in JSON format' in system_prompt:
        # Extract function definitions from system prompt
        func_start = system_prompt.find('[{')
        func_end = system_prompt.find('}]') + 2

        if func_start != -1 and func_end != -1:
            functions_json = system_prompt[func_start:func_end]

            # Clean up the system prompt (remove function definitions)
            clean_system = system_prompt[:func_start].strip()
            clean_system = clean_system.replace('Here is a list of functions in JSON format that you can invoke:', '').strip()

            formatted_text += f"{clean_system}\n"
            formatted_text += f"List of tools: <|tool_list_start|>{functions_json}<|tool_list_end|>"
        else:
            formatted_text += system_prompt
    else:
        formatted_text += system_prompt

    formatted_text += "<|im_end|>\n"

    # Process conversations
    for turn in conversations:
        role = turn['from']
        content = turn['value']

        if role == 'user':
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"

        elif role == 'assistant':
            formatted_text += "<|im_start|>assistant\n"

            # Check if this is a function call (starts with [)
            if content.startswith('[') and content.endswith(']'):
                formatted_text += f"<|tool_call_start|>{content}<|tool_call_end|>\n"
            else:
                formatted_text += f"{content}"

            formatted_text += "<|im_end|>\n"

        elif role == 'tool':
            formatted_text += f"<|im_start|>tool\n<|tool_response_start|>{content}<|tool_response_end|><|im_end|>\n"

    return {"text": formatted_text}

# Load and convert dataset
print("📥 Loading original dataset...")
train_dataset = load_dataset("Team-ACE/ToolACE", split="train[:80%]")
eval_dataset = load_dataset("Team-ACE/ToolACE", split="train[80%:]")

print("🔄 Converting to LFM2 format...")
train_dataset_lfm2 = train_dataset.map(convert_to_lfm2_format, remove_columns=train_dataset.column_names)
eval_dataset_lfm2 = eval_dataset.map(convert_to_lfm2_format, remove_columns=eval_dataset.column_names)

print("✅ Conversion complete!")
print(f"   📚 Train samples: {len(train_dataset_lfm2)}")
print(f"   🧪 Eval samples: {len(eval_dataset_lfm2)}")

# Show example
print(f"\n📝 Converted Sample:")
print(train_dataset_lfm2[0]['text'] + "...")

from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="./lfm2-sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=100,
    warmup_ratio=0.2,
    logging_steps=10,

    #save_strategy="epoch",
    # eval_strategy="epoch",

    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=5000,

    load_best_model_at_end=True,
    report_to='tensorboard',
    bf16=True # <- not all colab GPUs support bf16
)

print("🏗️  Creating SFT trainer...")
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset_lfm2,
    eval_dataset=eval_dataset_lfm2,
    processing_class=tokenizer,
)

print("\n🚀 Starting SFT training...")
sft_trainer.train()

print("🎉 SFT training completed!")

sft_trainer.save_model()
print(f"💾 SFT model saved to: {sft_config.output_dir}")
