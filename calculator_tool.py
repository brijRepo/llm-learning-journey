import ollama
import json
import re

def calculator(operation, a, b):
    if operation == "add":
        return a+b
    elif operation == "subtract":
        return a-b
    elif operation == "multiply":
        return a*b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        else:
            return a/b
    else:
        return f"Error: Unknown operation '{operation}'"
    
def chat_with_calculator(user_message):

    system_prompt = """You are a helpful assistant with access to a calculator tool.

    When the user asks a math question, use this format to call the calculator:

    CALCULATOR_CALL: {"operation": "add|subtract|multiply|divide", "a": number, "b": number}

    For example:
    - User asks "What is 5 + 3?"
    - You respond: CALCULATOR_CALL: {"operation": "add", "a": 5, "b": 3}

    After calling the calculator, I will give you the result, then you can answer the user.

    If the question is not math-related, just answer normally without the calculator."""


    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ]
    )

    ai_response = response['message']['content']

    if "CALCULATOR_CALL:" in ai_response:
        print("[AI is using calculator...]")

        match = re.search(r'CALCULATOR_CALL:\s*(\{.*?})', ai_response)
        if match:
            tool_call = json.loads(match.group(1))

            result = calculator(
                tool_call['operation'],
                tool_call['a'],
                tool_call['b']
            )

            print(f"[Calculator result: {result}]")

            final_response = ollama.chat(
                model='llama3.2:3b',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message},
                    {'role': 'assistant', 'content': ai_response},
                    {'role': 'user', 'content': f"Calculator result: {result}. Now answer the user's question."}
                ]
            )

            return final_response['message']['content']
        
    return ai_response


def main():
    print("Chat with Calculator Tool")
    print("=" * 60)
    print("Ask math questions and I'll use my calculator!\n")

    test_questions = [
        "What is 12345 multiplied by 67890?",
        "Calculate 100 divided by 5",
        "What's 999 plus 1?",
        "What's the weather like?",  # Non-math question
    ]
    
    for question in test_questions:
        print(f"You: {question}")
        answer = chat_with_calculator(question)
        print(f"AI: {answer}\n")
        print("-" * 60 + "\n")

        
if __name__ == "__main__":
    main()
    