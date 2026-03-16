# Step 1: Import required libraries
import os
from openai import OpenAI
from dotenv import load_dotenv

# Step 2: Load environment variables from .env file
load_dotenv()

# Step 3: Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Step 4: Check if API key exists
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    print("Please check your .env file and make sure the key is correct")
    exit()

print("API Key loaded successfully!")

# Step 5: Initialize Anthropic client
client = OpenAI(api_key = api_key)

# Step 6: Make API call
print("\nSending messages to OpenAI...")
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    max_tokens = 100,
    messages = [
        {
            "role": "User",
            "content": "Explain what an LLM is in one sentence."
        }
    ]
)

# Step 7: Print OpenAI's response
print("\n" + "="*50)
print("Claude's Response:")
print("="*50)
print(response.choices[0].message.content)

# Step 8: Print usage information (important for cost tracking!)
print("\n" + "="*50)
print("Usage information:")
print("="*50)
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.input_tokens} + {response.usage.output_tokens}")

# Step 9: Calculate cost
# Pricing (as of Jan 2025): Input $3/M tokens, Output $15/M tokens
# Pricing: GPT-4o-mini is $0.15/1M input, $0.60/1M output
input_cost = (response.usage.prompt_tokens) * 0.15 / 1000000
output_cost = (response.usage.completion_tokens) * 0.60 / 1000000
total_cost = input_cost + output_cost

print(f"\nEstimated cost: ${total_cost}")