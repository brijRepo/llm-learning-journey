from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

documents = [
    "Python is a high-level programming language",
    "JavaScript is primarily used for web development",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand text",
    "Computer vision enables machines to interpret images",
    "The Eiffel Tower is a landmark in Paris, France",
    "The Great Wall of China is visible from space",
    "Mount Everest is the tallest mountain in the world",
    "The Amazon rainforest produces 20% of Earth's oxygen"
]

print("Indexing documents...")
print("=" * 60)

document_embeddings = model.encode(documents)

print(f"Indexed {len(documents)} documents")
print()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def search(query, top_k=3):
    query_embedding = model.encode([query])[0]

    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score in similarities[:top_k]:
        results.append((documents[i], score))
    
    return results


test_queries = [
    "What is machine learning?",
    "Tell me about programming languages",
    "Famous landmarks around the world",
    "How do computers understand language?",
    "Natural wonders and geography"
]

print("Search Results:")
print("=" * 60)

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    results = search(query, top_k=3)
    
    print("Top results:")
    for rank, (doc, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.3f}] {doc}")
    print()
    