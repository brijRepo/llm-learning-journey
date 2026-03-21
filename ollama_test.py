# Import the ollama library
import ollama

# Simple non-streaming call
print("\nTesting Ollama call...")

# Call the model
response = ollama.chat(
    model = "llama3.2:3b",
    messages = [
        {
            "role": "user",
            "content": "Say hello in one sentence"
        }
    ]
)

# Print response
print("Response:", response["message"]["content"])
