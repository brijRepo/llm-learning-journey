from sentence_transformers import SentenceTransformer
import numpy as np
import json
import ollama

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

KB_FILE = "knowledge_base.json"

def load_kb():
    with open(KB_FILE, 'r') as f:
        kb = json.load(f)
    return kb

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def retrieve_relevant_notes(query, top_k=3):
    kb = load_kb()
    query_embedding = embedding_model.encode([query])[0]

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
    
def answer_question(question):
    relevant_notes = retrieve_relevant_notes(question, top_k=3)

    if not relevant_notes:
        return "I don't have any notes to answer this question."
    
    context = "\n\n".join([
        f"Note: {result['note']['title']}\n{result['note']['content']}"
        for result in relevant_notes
    ])

    prompt = f"""Based on these notes from my knowledge base, answer the question.
    If the notes don't contain enough information, say so.

    Notes:
    {context}

    Question: {question}

    Answer based on the notes above:"""

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']

def main():
    print("Knowledge Base Q&A (RAG System)")
    print("=" * 70)
    print("Ask questions about your notes!\n")

    while True:
        question = input("Question (or 'quit'): ")
        
        if question.lower() == "quit":
            break
        
        if not question.strip():
            continue
        
        print("\nSearching notes...")

        relevant = retrieve_relevant_notes(question, top_k=3)
        print(f"Retrieved {len(relevant)} relevant notes:")
        for i, result in enumerate(relevant, 1):
            print(f"  {i}. [{result['score']:.3f}] {result['note']['title']}")
        
        print("\nGenerating answer...\n")

        answer = answer_question(question)
        
        print("Answer:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
        print()

if __name__ == "__main__":
    main()
    
    