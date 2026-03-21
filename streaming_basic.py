# Import the ollama library
import ollama

# Print header
print("Streaming Response Demo")
print("="*50)
print("\nQuestion: What is Python?\n")

# Start the response label
print("AI:", end="", flush=True)

# Make streaming API call
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain what Python is in 2-3 sentences."
        }
    ],
    stream=True
)

# Loop through the stream
for chunk in response:
    text_piece = chunk["message"]["content"]
    print(text_piece, end="", flush=True)

# Add newline at the end
print("\n")
print("="*50)
print("Streaming complete!")
