import ollama
from json_utils import extract_json
import json

def analyze_sentiment(text):

    prompt = f"""Analyze the sentiment of this text.

    Text: "{text}"

    Return JSON in this format:
    {{
        "sentiment": "positive" or "negative" or "neutral",
        "confidence": 0.0 to 1.0,
        "reasoning": "brief explanation"
    }}

    Classification guide:
    - Positive: Happy, satisfied, enthusiastic
    - Negative: Angry, disappointed, frustrated
    - Neutral: Factual, balanced, no emotion

    JSON:"""

    response = ollama.chat(
        model= "llama3.2:3b",
        messages= [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return extract_json(response["message"]["content"])


def main():
    test_reviews = [
        "This product is amazing! Best purchase ever.",
        "It's okay, nothing special.",
        "Terrible quality. Complete waste of money.",
        "The service was helpful but product arrived damaged.",
        "Learning LLM engineering is challenging but exciting!"
    ]

    print("Sentiment Analysis Results")
    print("=" * 70)

    for review in test_reviews:
        result = analyze_sentiment(review)

        if result:
            print(f"\nText: {review}")
            print(f"Sentiment: {result.get('sentiment', 'unknown').upper()}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Reasoning: {result.get('reasoning', 'N/A')}")
            print("-" * 70)

if __name__ == "__main__":
    main()
    