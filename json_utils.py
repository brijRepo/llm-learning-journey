import json
import re

def clean_json_response(text):

    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    match = re.search(r'\{.*\}', text, re.DOTALL)

    if match:
        return match.group()
    
    return text.strip()


def parse_json_safely(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Problematic text: {text[:200]}...")  # Show first 200 chars
        return None
    

def extract_json(llm_response):
    cleaned = clean_json_response(llm_response)
    return parse_json_safely(cleaned)


if __name__ == "__main__":
    test_cases = [
        '{"name": "Alice", "age": 30}',
        '```json\n{"name": "Bob"}\n```',
        'Sure! Here\'s the data: {"name": "Charlie"} Hope this helps!',
        '{name: Alice}'
    ]

    print("Testing JSON cleaning utilities:\n")
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test[:50]}...")
        result = extract_json(test)
        print(f"Result: {result}")
        print()
