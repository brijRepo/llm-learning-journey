import ollama
from json_utils import extract_json

def extract_contact_schema_based(text):
    """
    Extract contact info using schema-based prompting.
    
    Args:
        text (str): Text containing name and email
        
    Returns:
        dict: {"name": str, "email": str}
    """
    
    # Build the prompt
    prompt = f"""Extract the name and email from this text.

Text: "{text}"

Return ONLY valid JSON in this exact format (no other text):
{{
  "name": "full name here",
  "email": "email@example.com"
}}

JSON:"""
    
    # Call the LLM
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    # Extract and return JSON
    return extract_json(response['message']['content'])


def extract_contact_few_shot(text):
    """
    Extract contact info using few-shot prompting.
    
    Args:
        text (str): Text containing name and email
        
    Returns:
        dict: {"name": str, "email": str}
    """
    
    # Build prompt with examples
    prompt = f"""Extract name and email from text. Follow these examples:

Example 1:
Input: "Contact John at john@email.com"
Output: {{"name": "John", "email": "john@email.com"}}

Example 2:
Input: "My name is Alice Smith. Email: alice@company.org"
Output: {{"name": "Alice Smith", "email": "alice@company.org"}}

Example 3:
Input: "Reach out to bob@test.com for Bob"
Output: {{"name": "Bob", "email": "bob@test.com"}}

Now extract from this:
Input: "{text}"
Output:"""
    
    # Call the LLM
    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return extract_json(response['message']['content'])

if __name__ == "__main__":
    test_texts = [
        "Contact Jane Doe at jane.doe@company.com",
        "My name is Robert Smith, email rsmith@email.org",
        "Email alice@wonderland.com for Alice"
    ]
    
    print("PATTERN 1: Schema-Based Prompting")
    print("=" * 60)
    for text in test_texts:
        print(f"\nInput: {text}")
        result = extract_contact_schema_based(text)
        if result:
            print(f"Output: {result}")
            print(f"  Name: {result.get('name', 'N/A')}")
            print(f"  Email: {result.get('email', 'N/A')}")
    
    print("\n\n" + "=" * 60)
    print("PATTERN 2: Few-Shot Prompting")
    print("=" * 60)
    for text in test_texts:
        print(f"\nInput: {text}")
        result = extract_contact_few_shot(text)
        if result:
            print(f"Output: {result}")
            print(f"  Name: {result.get('name', 'N/A')}")
            print(f"  Email: {result.get('email', 'N/A')}")
