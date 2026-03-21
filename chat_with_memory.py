import ollama

def chat_with_history(messages, model='llama3.2:3b'):

    # Print AI label
    print("AI: ", end="", flush=True)
    
    # Store full response
    full_response = ""
    
    # Make streaming call with entire history
    stream = ollama.chat(
        model=model,
        messages=messages,  # Send ALL messages (not just latest)
        stream=True,
    )
    
    # Stream the response
    for chunk in stream:
        text_piece = chunk['message']['content']
        print(text_piece, end="", flush=True)
        full_response += text_piece
    
    print("\n")
    return full_response

def main():
    print("Chat with Memory (type 'quit' to exit)")
    print("="*50)
    print()
    
    # Initialize empty conversation
    conversation = []
    
    # Infinite loop for chat
    while True:
        
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to quit
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        # Add user message to conversation
        conversation.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get AI response
        response = chat_with_history(conversation)
        
        # Add AI response to conversation
        conversation.append({
            'role': 'assistant',
            'content': response
        })
        
        print()  # Blank line for readability

if __name__ == "__main__":
    main()