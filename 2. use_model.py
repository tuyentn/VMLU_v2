import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_deepseek_model(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    Load the DeepSeek-R1-Distill-Qwen-1.5B model and tokenizer.
    
    Args:
        model_path (str): HuggingFace model path or local path to the model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with lower precision for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision for efficiency
        device_map="auto"  # Automatically determine the best device mapping
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """
    Generate a response from the model based on the prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt (str): Input prompt text
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature (higher = more creative)
        
    Returns:
        str: The generated response
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return only the newly generated text (not including the prompt)
    return response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU may be slow. GPU is recommended for faster inference.")
    
    # Load the model and tokenizer
    model, tokenizer = setup_deepseek_model()
    
    # Interactive mode
    print("\nDeepSeek-R1-Distill-Qwen-1.5B Model Loaded! Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, user_input)
        print(f"\nDeepSeek: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()