import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import json


client = chromadb.PersistentClient(path="./chroma_db")

print("ChromaDB client initialized.")
print(f"Existing collections: {[c.name for c in client.list_collections()]}")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

COLLECTION_NAME = "llm_engineering_docs"

try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"description": "LLM engineering knowledge base for Day 7"}
)
print(f"Created collection: {COLLECTION_NAME}")


documents = [
    "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents at query time and injects them into the LLM prompt as grounding context.",
    "Temperature in LLMs controls the randomness of token sampling. A value of 0 is deterministic (greedy). Values above 1.0 produce highly varied outputs.",
    "Vector embeddings are dense numerical representations of text where semantic similarity corresponds to spatial proximity in high-dimensional space.",
    "Fine-tuning updates the weights of a pre-trained model on domain-specific data. It is expensive and static compared to RAG.",
    "LangChain is a framework for composing LLM applications. It provides abstractions for chains, agents, memory, and tool use.",
    "ChromaDB is a local vector database that stores embeddings and supports approximate nearest neighbor search. It requires no server setup.",
    "Cosine similarity measures the angle between two vectors. It is magnitude-invariant, making it ideal for comparing embeddings of different-length texts.",
    "The context window is the maximum number of tokens an LLM can process at once. Both input and output count against this limit.",
    "Prompt injection is an attack where malicious user input overrides developer instructions in the system prompt.",
    "The Transformer architecture uses stacked self-attention layers. Each layer allows every token to attend to every other token in the sequence.",
]

ids = [f"doc_{i:03d}" for i in range(len(documents))]

metadatas = [
    {"topic": "rag",           "difficulty": "beginner"},
    {"topic": "temperature",   "difficulty": "beginner"},
    {"topic": "embeddings",    "difficulty": "intermediate"},
    {"topic": "fine-tuning",   "difficulty": "intermediate"},
    {"topic": "langchain",     "difficulty": "beginner"},
    {"topic": "chromadb",      "difficulty": "beginner"},
    {"topic": "similarity",    "difficulty": "intermediate"},
    {"topic": "context-window","difficulty": "beginner"},
    {"topic": "security",      "difficulty": "intermediate"},
    {"topic": "transformers",  "difficulty": "advanced"},
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
print(f"\nAdded {len(documents)} documents to collection.")
print(f"Collection count: {collection.count()}")

print("\n=== SEMANTIC QUERY ===")

query_text = "How does RAG work and why is it better than fine-tuning?"

results = collection.query(
    query_texts=[query_text],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

print(f"Query: '{query_text}'\n")
for i, (doc, dist, meta) in enumerate(zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
)):
    similarity = 1 / (1 + dist)
    print(f"Rank {i+1} | Similarity: {similarity:.4f} | Topic: {meta['topic']}")
    print(f"  → {doc}\n")

print("=== FILTERED QUERY (beginners only) ===")

filtered_results = collection.query(
    query_texts=["explain vector search"],
    n_results=3,
    where={"difficulty": "beginner"},
    include=["documents", "metadatas"]
)

for doc, meta in zip(
    filtered_results["documents"][0],
    filtered_results["metadatas"][0]
):
    print(f"  [{meta['difficulty']}] {meta['topic']}: {doc[:80]}...")


print("\n=== UPDATE A DOCUMENT ===")

collection.update(
    ids=["doc_000"],
    documents=["RAG (Retrieval Augmented Generation) retrieves relevant document chunks at query time using vector similarity search, injects them as context into the LLM prompt, and generates grounded, citation-backed answers."],
    metadatas=[{"topic": "rag", "difficulty": "intermediate"}]
)
print("Updated doc_000 with improved RAG definition.")

print("\n=== DELETE A DOCUMENT ===")
collection.delete(ids=["doc_009"])
print(f"Deleted doc_009. New count: {collection.count()}")

print("\n=== COLLECTION PEEK ===")
peek = collection.peek(limit=3)

for doc_id, doc_text in zip(peek["ids"], peek["documents"]):
    print(f"  {doc_id}: {doc_text[:70]}...")
    