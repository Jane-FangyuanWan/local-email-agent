import json
import torch
import os
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ================= Configuration Area =================
# For Mac M-series, larger models might run out of memory, so we use a very fast
# ungated model (Qwen 0.5B) here so you don't have to deal with HuggingFace login keys!
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
# ======================================================

def main():
    print("🚀 Initializing Knowledge Distillation & Fine-Tuning Script...")

    # 1. Load the synthetic dataset
    print(f"Loading synthetic dataset from dataset.json...")
    with open("dataset.json", "r") as f:
        raw_data = json.load(f)

    # 2. Format the data for Instruction Tuning
    # We must format the text exactly how the model will see it during inference.
    formatted_data = []
    for item in raw_data:
        # We form a single prompt containing both the input and the expected JSON output.
        prompt = f"""You are an advanced email assistant. Analyze this email and output a specific JSON schema.
        
Input Email:
{item['input_email']}

Output:
{json.dumps(item['output_json'])}"""
        
        formatted_data.append({"text": prompt})

    # Convert to HuggingFace Dataset format
    hf_dataset = Dataset.from_list(formatted_data)
    print(f"✅ Loaded {len(hf_dataset)} training examples.")

    # 3. Hardware check: Apple Silicon (MPS), Nvidia (CUDA), or fallback CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("💻 Apple Silicon (MPS) detected! Using GPU acceleration.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("💻 NVIDIA GPU (CUDA) detected! Using GPU acceleration.")
    else:
        device = "cpu"
        print("⚠️ Warning: No GPU detected, falling back to slow CPU.")

    # 4. Load the Tokenizer and Base Model
    print(f"Loading base model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model efficiently (fp16 on GPU, fp32 on CPU)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    # 5. Configure LoRA (Low-Rank Adaptation)
    # This freezes 99% of the model and only trains a tiny "adapter" network,
    # saving massive amounts of RAM and time.
    print("Configuring LoRA (PEFT)...")
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Target Attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Setup Training Parameters
    training_args = SFTConfig(
        output_dir="./local-email-agent-outputs",
        per_device_train_batch_size=2, # Small batch size to fit in Mac/Laptop RAM
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=3, # 3 epochs across 100 emails is a solid start
        optim="adamw_torch",
        save_strategy="no", # We only want the final model for this small project
        fp16=(device=="cuda"), # Use true mixed precision only on Nvidia, MPS uses model dtype
    )

    # 7. Initialize SFTTrainer (Supervised Fine-Tuning)
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        args=training_args,
    )

    # 8. Start the Training
    print("🔥 Beginning Fine-Tuning... (This may take a few minutes to an hour depending on your hardware)")
    trainer.train()

    # 9. Save the Final Adapter Model
    save_path = "my_custom_email_adapter"
    print(f"✅ Training complete. Saving the adapted model weights to {save_path}/...")
    
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("\n🎉 Success! You have mathematically adapted the AI to your specific task.")
    print(f"Your fine-tuned LoRA weights are saved in: {os.path.abspath(save_path)}")
    print("To use this model in your backend.py, simply load the base model and apply the adapter using PEFT!")

if __name__ == "__main__":
    main()
