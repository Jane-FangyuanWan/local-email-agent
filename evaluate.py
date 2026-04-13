import json
import time
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

def evaluate_llama(emails):
    print("\n--- Evaluating Llama 3.2 (Zero-Shot via Ollama) ---")
    start_time = time.time()
    correct_category = 0
    correct_action = 0
    total = len(emails)
    
    for item in emails:
        prompt = f"""
        Analyze this email.
        {item['input_email']}
        """
        # Define Pydantic schema for strict JSON
        class EmailAnalysis:
            category: str
            action: str
            draft: str
            
            @classmethod
            def model_json_schema(cls):
                return {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "action": {"type": "string"},
                        "draft": {"type": "string"}
                    },
                    "required": ["category", "action", "draft"]
                }
        try:
            response = ollama.chat(
                model='llama3.2', 
                messages=[{'role': 'user', 'content': prompt}], 
                format=EmailAnalysis.model_json_schema()
            )
            content = json.loads(response['message']['content'])
            
            if content.get("category") == item['output_json']['category']:
                correct_category += 1
            if content.get("action") == item['output_json']['action']:
                correct_action += 1
        except Exception as e:
            print(f"Error: {e}")
            pass
            
    latency = (time.time() - start_time) / total
    cat_acc = correct_category / total
    act_acc = correct_action / total
    print(f"Average Latency: {latency:.2f} seconds/email")
    print(f"Category Accuracy: {cat_acc*100:.1f}%")
    print(f"Action Accuracy: {act_acc*100:.1f}%")
    return latency, cat_acc, act_acc

def evaluate_qwen_lora(emails):
    print("\n--- Evaluating Qwen-0.5B + LoRA (Fine-Tuned) ---")
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = "./my_custom_email_adapter"
    
    try:
        config = PeftConfig.from_pretrained(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Could not load LoRA model. Details: {e}")
        return 0, 0, 0
        
    start_time = time.time()
    correct_category = 0
    correct_action = 0
    total = len(emails)
    
    for item in emails:
        prompt = f"""You are an advanced email assistant. Analyze this email and output a specific JSON schema.
        
Input Email:
{item['input_email']}

Output:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        
        output_str = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            # Attempt to parse output JSON
            content = json.loads(output_str)
            if content.get("category") == item['output_json']['category']:
                correct_category += 1
            if content.get("action") == item['output_json']['action']:
                correct_action += 1
        except:
            pass
            
    latency = (time.time() - start_time) / total
    cat_acc = correct_category / total
    act_acc = correct_action / total
    print(f"Average Latency: {latency:.2f} seconds/email")
    print(f"Category Accuracy: {cat_acc*100:.1f}%")
    print(f"Action Accuracy: {act_acc*100:.1f}%")
    return latency, cat_acc, act_acc

if __name__ == "__main__":
    print("Loading dataset...")
    # Evaluate on a sample of 20 emails to save time
    with open("dataset.json", "r") as f:
        data = json.load(f)[:20]
        
    llama_lat, llama_cat, llama_act = evaluate_llama(data)
    qwen_lat, qwen_cat, qwen_act = evaluate_qwen_lora(data)
    
    print("\n" + "="*40)
    print("RESULTS SUMMARY FOR PAPER")
    print("="*40)
    print(f"Llama 3.2 Zero-Shot: Latency {llama_lat:.2f}s, Cat Acc {llama_cat*100:.1f}%, Act Acc {llama_act*100:.1f}%")
    print(f"Qwen + LoRA FT: Latency {qwen_lat:.2f}s, Cat Acc {qwen_cat*100:.1f}%, Act Acc {qwen_act*100:.1f}%")
