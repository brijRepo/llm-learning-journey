from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from datetime import datetime
import pickle  # For caching embeddings

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")

KB_FILE = "knowledge_base.json"
CACHE_FILE = "embedding_cache.pkl"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def get_embedding_cached(text):
    cache = load_cache()

    if text in cache:
        print(f"Cache hit: {text[:30]}...")
        return np.array(cache[text])
    
    print(f"Embedding: {text[:30]}...")
    embedding = model.encode([text])[0]

    cache[text] = embedding.tolist()
    save_cache(cache)
    
    return embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def batch_search(queries, documents, top_k=3):
    print(f"\nBatch processing {len(queries)} queries...")

    query_embeddings = model.encode(queries)

    doc_embeddings = model.encode(documents)

    all_results = {}

    for i, query in enumerate(queries):
        query_emb = query_embeddings[i]

        similarities = []
        for j, doc_emb in enumerate(doc_embeddings):
            sim = cosine_similarity(query_emb, doc_emb)
            similarities.append((j, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = [
            {"document": documents[idx], "score": score}
            for idx, score in similarities[:top_k]
        ]
        
        all_results[query] = results
    
    return all_results


def demo_caching():
    texts = [
        "Python programming language",
        "Machine learning algorithms",
        "Natural language processing",
        "Python programming language",  # Duplicate - should hit cache
        "Deep learning neural networks",
        "Machine learning algorithms"  # Duplicate - should hit cache
    ]

    print("Demonstrating Embedding Cache:")
    print("=" * 70)

    for text in texts:
        embedding = get_embedding_cached(text)
        print(f"  Got embedding: shape {embedding.shape}")
    
    print("\n" + "=" * 70)
    print("Notice: Duplicates use cache (much faster)!")
    print()

def demo_batch_search():
    queries = [
        "Python programming",
        "Machine learning concepts",
        "Natural language understanding"
    ]

    documents = [
        "Python is a high-level programming language",
        "Machine learning is a branch of AI",
        "Deep learning uses neural networks",
        "NLP helps computers understand human language",
        "Python is great for data science",
        "AI and ML are transforming technology"
    ]

    print("Demonstrating Batch Search:")
    print("=" * 70)
    
    results = batch_search(queries, documents, top_k=2)
    
    for query, query_results in results.items():
        print(f"\nQuery: {query}")
        print("-" * 70)
        for i, result in enumerate(query_results, 1):
            print(f"  {i}. [{result['score']:.3f}] {result['document']}")


def main():
    print("Optimized Knowledge Base Features\n")

    demo_caching()

    demo_batch_search()

    cache = load_cache()
    print("\n" + "=" * 70)
    print(f"Cache Statistics:")
    print(f"Total cached embeddings: {len(cache)}")
    print(f"Cache file size: {os.path.getsize(CACHE_FILE) / 1024:.2f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
    