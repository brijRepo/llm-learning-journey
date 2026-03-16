#import
import subprocess

#method
def call_local_llm(prompt, model = "llama3.2:3b"):

    print(f"Sending message to {model}")

    # Prepare the command
    cmd = [
        "ollama",
        "run",
        model,
        prompt
    ]

    # Run the command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    response = result.stdout.strip()

    return response

    
# Main execution
if __name__ == "__main__":
    print("Local LLM API Call")
    print("="*50)

    # Your prompt
    prompt = "Explain what an LLM is in one sentence."

    # Get response
    response = call_local_llm(prompt)

    # Print response
    print("\n" + "="*50)
    print("LLM Response:")
    print("="*50)
    print(response)

    # Note about cost
    print("\n" + "="*50)
    print("Cost Information:")
    print("="*50)
    print("Cost: $0.00 (Running locally, completely free!)")
    print("Model: Llama 3.2 3B")
    print("Running on: Your computer")
