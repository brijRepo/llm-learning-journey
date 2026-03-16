import subprocess

def estimate_tokens(text):
    """Rough estimation: 1 token ≈ 0.75 words"""
    words = text.split()
    estimated_tokens = int(len(words) / 0.75)
    return estimated_tokens

def call_local_llm(prompt, model="llama3.2:3b"):
    """Call local Ollama model"""
    cmd = ["ollama", "run", model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    print("Local LLM Cost Estimator")
    print("="*50)
    
    # Get user input
    user_message = input("\nEnter your message to LLM: ")
    
    # Estimate input tokens
    input_tokens = estimate_tokens(user_message)
    
    print("\n" + "="*50)
    print("ESTIMATE:")
    print("="*50)
    print(f"Input tokens: ~{input_tokens}")
    print(f"Cost: $0.00 (local model)")
    
    # Make actual call
    print("\nCalling local LLM...")
    response = call_local_llm(user_message)
    
    # Estimate output tokens
    output_tokens = estimate_tokens(response)
    
    print("\n" + "="*50)
    print("ACTUAL:")
    print("="*50)
    print(f"Input tokens: ~{input_tokens}")
    print(f"Output tokens: ~{output_tokens}")
    print(f"Total tokens: ~{input_tokens + output_tokens}")
    print(f"Cost: $0.00")
    
    print("\n" + "="*50)
    print("RESPONSE:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    main()
    