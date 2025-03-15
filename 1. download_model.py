from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_model(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", save_dir="./deepseek-model"):
    """
    Download the model and tokenizer for offline use.
    
    Args:
        model_id (str): HuggingFace model ID
        save_dir (str): Directory to save the model
    """
    print(f"Downloading model {model_id}...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)
    
    # Download model
    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(save_dir)
    
    print(f"Model and tokenizer downloaded successfully to {save_dir}")
    print("You can now use this local path in the main script.")

if __name__ == "__main__":
    download_model()