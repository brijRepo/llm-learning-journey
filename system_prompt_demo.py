import ollama

def chat_with_system_prompt(system_prompt, user_prompt, model = "llama3.2:3b"):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = ollama.chat(
        model = model,
        messages = messages,
        # stream = True
    )

    return response["message"]["content"]

def main():

    print("System Prompt Demo")
    print("="*50)
    
    # Same question, different personas
    question = "Explain what a variable is in programming."
    
    # Test 1: Default (no system prompt)
    print("\n1. DEFAULT (no system prompt)")
    print("-"*50)
    response = chat_with_system_prompt("", question)
    print(response)
    
    # Test 2: Pirate persona
    print("\n2. PIRATE PERSONA")
    print("-"*50)
    pirate_prompt = "You are a pirate who explains programming. Use pirate language and say 'arr' often."
    response = chat_with_system_prompt(pirate_prompt, question)
    print(response)
    
    # Test 3: Simple explainer
    print("\n3. SIMPLE EXPLAINER (for kids)")
    print("-"*50)
    simple_prompt = "You explain programming to 8-year-olds. Use analogies with toys and games. Maximum 2 sentences."
    response = chat_with_system_prompt(simple_prompt, question)
    print(response)
    
    # Test 4: Expert consultant
    print("\n4. SENIOR ENGINEER")
    print("-"*50)
    expert_prompt = "You are a senior software engineer. Be concise and technical. Include a code example."
    response = chat_with_system_prompt(expert_prompt, question)
    print(response)

    return response


if __name__ == "__main__":
    main()