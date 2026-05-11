from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Embedding model loaded.")
print(f"Output dimensions: {model.get_sentence_embedding_dimension()}")

sentence = "What is retrieval augmented generation?"

embedding = model.encode(sentence)

print(f"\n=== Single Embedding ===")
print(f"Text: '{sentence}'")
print(f"Type: {type(embedding)}")
print(f"Shape: {embedding.shape}")

print(f"First 5 values: {embedding[:5].round(4)}")
print(f"Min value: {embedding.min():.4f}")
print(f"Max value: {embedding.max():.4f}")


sentences = [
    "RAG stands for Retrieval Augmented Generation.",
    "Retrieval augmented generation combines search with LLMs.",
    "RAG retrieves documents and injects them into the prompt.",

    "The monsoon season in Pune brings heavy rainfall.",
    "Maharashtra receives most of its rain between June and September.",

    "How do transformers use attention?",
    "Explain the attention mechanism in transformer models.",
]

embeddings = model.encode(sentences)

print(f"\n=== Batch Embeddings ===")
print(f"Input: {len(sentences)} sentences")
print(f"Output shape: {embeddings.shape}")

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))

print(f"\n=== Cosine Similarity Matrix ===")
print(f"{'':>5}", end="")
labels = [f"S{i+1}" for i in range(len(sentences))]
print("  ".join(f"{l:>6}" for l in labels))

for i, (sent_i, emb_i) in enumerate(zip(sentences, embeddings)):
    print(f"S{i+1}: ", end="")
    for j, emb_j in enumerate(embeddings):
        sim = cosine_similarity(emb_i, emb_j)
        print(f"{sim:>6.3f}  ", end="")
    print(f"  ← {sent_i[:40]}...")

def find_most_similar(query: str, corpus: List[str], top_k: int = 3) -> List[tuple]:
    query_embedding = model.encode(query)
    corpus_embeddings = model.encode(corpus)

    similarities = []
    for i, corp_emb in enumerate(corpus_embeddings):
        sim = cosine_similarity(query_embedding, corp_emb)
        similarities.append((sim, corpus[i]))

    similarities.sort(key=lambda x: x[0], reverse=True)

    return similarities[:top_k]

query = "How does RAG work?"
print(f"\n=== Similarity Search ===")
print(f"Query: '{query}'\n")
results = find_most_similar(query, sentences, top_k=3)

for rank, (score, text) in enumerate(results, 1):
    print(f"Rank {rank} | Score: {score:.4f} | Text: {text}")
    