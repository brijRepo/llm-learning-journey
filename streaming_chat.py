import ollama

def stream_chat(user_message, model="llama3.2:3b"):
    # Start printing the AI label
    print("AI:", end="", flush=True)

    # Variable to store complete response
    full_response = ""

    # Make streaming API call
    stream = ollama.chat(
        model = model,
        messages= [
            {
                "role": "user",
                "content": "user_message"
            }
        ],
        stream=True
    )

    # Loop through stream
    for chunk in stream:
        text_piece = chunk["message"]["content"]
        print(text_piece, end="", flush=True)
        full_response += text_piece

    # Add newline after response
    print("\n")

    # Return complete response
    return full_response

# Main execution
if __name__ == "__main__":
    print("Streaming Chat Demo")
    print("="*50)

    # Get user input
    user_input = input("\nYou:")

    # Get and print response
    response = stream_chat(user_input)

    # Could use response for other things
    print(f"[Response was {len(response)} characters long]")
