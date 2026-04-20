from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from datetime import datetime

print("Loading Embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

KB_FILE = 'knowledge_base.json'

def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, 'r') as f:
            return json.load(f)
    return {"notes": []}

def save_kb(kb):
    with open(KB_FILE, 'w') as f:
        json.dump(kb, f, indent=2)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def add_note(title, content, tags=None):
    kb = load_kb()
    text_to_embed = f"{title}.{content}"
    embedding = model.encode([text_to_embed])[0]
    note = {
        "id": len(kb["notes"]) + 1,
        "title": title,
        "content": content,
        "tags": tags or [],
        "created": datetime.now().isoformat(),
        "embedding": embedding.tolist()  # Convert numpy to list for JSON
    }
    kb["notes"].append(note)
    save_kb(kb)
    print(f"Added note #{note['id']}: {title}")
    return note["id"]

def search_notes(query, top_k=5):
    kb = load_kb()
    if not kb["notes"]:
        return []
    
    query_embedding = model.encode([query])[0]

    results = []
    for note in kb["notes"]:
        note_embedding = np.array(note["embedding"])
        similarity = cosine_similarity(query_embedding, note_embedding)
        
        results.append({
            "note": note,
            "score": similarity
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]

def list_all_notes():
    kb = load_kb()
    
    if not kb["notes"]:
        print("No notes yet!\n")
        return
    
    print(f"\nTotal notes: {len(kb['notes'])}\n")
    print("=" * 70)

    for note in kb["notes"]:
        print(f"\n#{note['id']}: {note['title']}")
        print(f"Created: {note['created']}")
        print(f"Tags: {', '.join(note['tags']) if note['tags'] else 'None'}")
        print(f"Content: {note['content'][:100]}...")
        print("-" * 70)


def get_note(note_id):
    kb = load_kb()
    
    for note in kb["notes"]:
        if note["id"] == note_id:
            return note
    
    return None


def delete_note(note_id):
    kb = load_kb()
    
    kb["notes"] = [n for n in kb["notes"] if n["id"] != note_id]
    save_kb(kb)
    
    print(f"Deleted note #{note_id}")



def main():
    print("Personal Knowledge Base with Semantic Search")
    print("=" * 70)
    print("Commands: add, search, list, view, delete, quit\n")

    while True:
        command = input("Command: ").lower().strip()

        if command == "quit":
            print("Goodbye!")
            break
        
        elif command == "add":
            print("\nAdd New Note")
            print("-" * 70)

            title = input("Title: ")
            print("Content (type END on new line when done):")
            
            lines = []
            while True:
                line = input()
                if line == "END":
                    break
                lines.append(line)
            
            content = "\n".join(lines)
            
            tags_input = input("Tags (comma-separated, optional): ")
            tags = [t.strip() for t in tags_input.split(",")] if tags_input else []
            
            add_note(title, content, tags)
            print()

        elif command == "search":
            query = input("\nSearch query: ")
            
            results = search_notes(query, top_k=5)
            
            if not results:
                print("No notes found!\n")
            else:
                print(f"\nTop {len(results)} results:")
                print("=" * 70)
                
                for i, result in enumerate(results, 1):
                    note = result["note"]
                    score = result["score"]
                    
                    print(f"\n{i}. [{score:.3f}] #{note['id']}: {note['title']}")
                    print(f"   {note['content'][:150]}...")
                    print(f"   Tags: {', '.join(note['tags']) if note['tags'] else 'None'}")
                
                print("\n" + "=" * 70 + "\n")

        elif command == "list":
            list_all_notes()
            print()

        elif command == "view":
            note_id = int(input("\nNote ID: "))
            note = get_note(note_id)
            
            if note:
                print("\n" + "=" * 70)
                print(f"#{note['id']}: {note['title']}")
                print("=" * 70)
                print(f"Created: {note['created']}")
                print(f"Tags: {', '.join(note['tags']) if note['tags'] else 'None'}")
                print(f"\n{note['content']}")
                print("=" * 70 + "\n")
            else:
                print(f"Note #{note_id} not found!\n")

        elif command == "delete":
            note_id = int(input("\nNote ID to delete: "))
            
            note = get_note(note_id)
            if note:
                confirm = input(f"Delete '{note['title']}'? (yes/no): ")
                if confirm.lower() == "yes":
                    delete_note(note_id)
            else:
                print(f"Note #{note_id} not found!")
            
            print()
        
        else:
            print("Unknown command. Try: add, search, list, view, delete, quit\n")

if __name__ == "__main__":
    main()

