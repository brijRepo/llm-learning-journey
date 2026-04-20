from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

texts = [
    "I love programming in Python",
    "Python is my favorite programming language",
    "I enjoy cooking Italian food",
    "The weather is nice today",
    "Machine learning is fascinating"
]

print("Generating embeddings...")
print("=" * 60)

embeddings = model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")
print()
print("First embedding (first 10 numbers):")
print(embeddings[0][:10])
print()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    similarity = dot_product/(norm1 * norm2)

    return similarity

print("Similarity Matrix:")
print("=" * 60)
print("(1.0 = identical, 0.0 = unrelated)\n")

for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        
        print(f"{i} vs {j}: {similarity:.3f}")
        print(f"  '{text1}'")
        print(f"  '{text2}'")
        print()
