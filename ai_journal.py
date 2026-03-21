import ollama
import json
from datetime import datetime

JOURNAL_SYSTEM_PROMPT = """You are a thoughtful journaling companion. When the user shares their daily reflection:
1. Acknowledge their feelings with empathy
2. Ask one insightful follow-up question
3. Offer brief encouragement or perspective

Keep responses warm and concise (3-4 sentences)."""


def get_ai_reflection(journal_entry, model='llama3.2:3b'):
    messages = [
        {
            "role": "system",
            "content": JOURNAL_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": journal_entry
        }
    ]

    response = ollama.chat(
        model = model,
        messages=messages
    )

    return response["message"]["content"]


def save_entry(entry_text, ai_reflection):
    filename = "journal_entries.json"

    try:
        with open(filename, "r") as f:
            data = json.load(f)

    except FileNotFoundError:
        data = {"entries": []}

        new_entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry": entry_text,
            "ai_reflection": ai_reflection
        }
    data["entries"].append(new_entry)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def view_past_entries(n=5):

    filename = "journal_entries.json"
    
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("No entries yet! Write your first one.\n")
        return
    
    entries = data["entries"][-n:]
    print(f"\n=== Last {len(entries)} Entries ===\n")

    for entry in entries:
        print(f"Date: {entry['date']}")
        print(f"Entry: {entry['entry'][:100]}...")
        print(f"AI: {entry['ai_reflection'][:100]}...")
        print("-" * 50)
        print()
        

def main():
    print("AI Journal - Daily Reflection Tool")
    print("="*50)
    print("Commands: 'write' (new entry), 'read' (past entries), 'quit'")
    print()

    while True:
        command = input("Command: ").lower()

        if command == "quit":
            print("GoodBye!")
            break

        elif command == "write":
            print("\nWrite your journal entry.")
            print("(Press Enter twice when done)")
            print("-" * 50)

            lines = []
            empty_count = 0

            while True:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                else:
                    empty_count = 0
                    lines.append(line)

            entry_text = "\n".join(lines)

            if not entry_text.split():
                print("Empty entry. Cancelled.\n")
                continue

            print("\nAI is reflecting...")
            ai_response = get_ai_reflection(entry_text)

            print("\n" + "="*50)
            print("AI Reflection:")
            print("="*50)
            print(ai_response)
            print()

            save_entry(entry_text, ai_response)
            print("Entry saved!\n")

        elif command == "read":
            view_past_entries(n=5)

        else:
            print("Incorrect command!")

if __name__ == "__main__":
    main()
