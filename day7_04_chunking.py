from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

document = """
Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses by grounding them in retrieved external knowledge. The core motivation is to solve the hallucination problem: LLMs generate plausible but sometimes incorrect text because they can only recall information compressed into their weights during training.

The RAG pipeline has two main phases. In the indexing phase, documents are loaded, split into chunks, converted to embedding vectors using an embedding model, and stored in a vector database. This happens once and is updated whenever the knowledge base changes.

In the retrieval phase, the user's query is embedded using the same model. The vector database performs a similarity search to find the top-k most relevant chunks. These chunks are injected into the LLM's prompt as context, and the model generates a grounded answer.

Chunking strategy is critical to RAG performance. If chunks are too large, they contain too much irrelevant content that dilutes the signal. If chunks are too small, they lose the surrounding context needed to understand the chunk's meaning. The typical sweet spot is 256 to 512 tokens with an overlap of 10 to 20 percent between consecutive chunks.

Overlap ensures that sentences at chunk boundaries are not lost. Without overlap, a key sentence that straddles two chunks might never appear in full in any retrieved result. With overlap, both chunks contain the sentence, so at least one will be retrieved.

Advanced chunking strategies include semantic chunking, which uses embedding similarity to find natural topic boundaries, and sentence-window retrieval, which indexes individual sentences but returns surrounding sentences as context when a sentence is retrieved.
"""


print("=== STRATEGY 1: Fixed-Size (CharacterTextSplitter) ===\n")

fixed_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,
    chunk_overlap=0,
    length_function=len,
)

fixed_chunks = fixed_splitter.split_text(document)
print(f"Number of chunks: {len(fixed_chunks)}")
print(f"Chunk sizes: {[len(c) for c in fixed_chunks]}")
for i, chunk in enumerate(fixed_chunks):
    print(f"\n[Chunk {i+1}] ({len(chunk)} chars)\n{chunk[:200]}...")

print("\n\n=== STRATEGY 2: Recursive + Overlap (RecursiveCharacterTextSplitter) ===\n")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

recursive_chunks = recursive_splitter.split_text(document)
print(f"Number of chunks: {len(recursive_chunks)}")
print(f"Chunk sizes: {[len(c) for c in recursive_chunks]}")

for i, chunk in enumerate(recursive_chunks):
    print(f"\n[Chunk {i+1}] ({len(chunk)} chars)\n{chunk}")

print("\n=== OVERLAP VERIFICATION ===")
if len(recursive_chunks) >= 2:
    print(f"End of Chunk 1:   ...{recursive_chunks[0][-100:]}")
    print(f"Start of Chunk 2: {recursive_chunks[1][:100]}...")

print("\n\n=== STRATEGY 3: Semantic Chunking (custom) ===\n")

def semantic_chunk(text: str, threshold: float = 0.75) -> List[str]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]

    if len(sentences) <= 1:
        return sentences
    
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        prev_emb = embeddings[i-1]
        curr_emb = embeddings[i]

        dot = np.dot(prev_emb, curr_emb)
        sim = float(dot / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb)))

        if sim < threshold:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(". ".join(current_chunk))

    return chunks

# from typing import List
semantic_chunks = semantic_chunk(document, threshold=0.70)
print(f"Number of semantic chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks):
    print(f"\n[Semantic Chunk {i+1}]\n{chunk}")

print("""
=== CHUNKING STRATEGY COMPARISON ===

Strategy              | When to Use
----------------------|------------------------------------------
Fixed-Size            | Quick prototypes, uniform documents
Recursive + Overlap   | General purpose — best default choice
Semantic              | High-precision RAG, topic-diverse docs
Sentence Window       | When surrounding context is critical
""")
