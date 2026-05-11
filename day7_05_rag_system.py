import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import hashlib

KNOWLEDGE_BASE = [
    {
        "title": "Introduction to RAG",
        "content": """Retrieval Augmented Generation (RAG) is a technique that combines
information retrieval with language generation. Instead of relying solely on
the knowledge baked into model weights during training, RAG dynamically fetches
relevant information at inference time. The retrieved content is injected into
the prompt as context, allowing the model to generate answers grounded in
real, up-to-date information. RAG dramatically reduces hallucination because
the model can reference actual source documents rather than reconstructing
facts from compressed training data."""
    },
    {
        "title": "RAG Pipeline Architecture",
        "content": """The RAG pipeline consists of two phases. During indexing,
documents are loaded and split into chunks using a text splitter. Each chunk is
converted to an embedding vector using an embedding model like all-MiniLM-L6-v2.
These vectors are stored in a vector database such as ChromaDB or Pinecone
along with the original text and metadata. During retrieval, the user's query
is embedded using the same model. A similarity search finds the top-k most
relevant chunks. These chunks are formatted as context and injected into the
LLM prompt. The language model reads the context and generates a grounded answer."""
    },
    {
        "title": "Chunking Best Practices",
        "content": """Chunking is the process of splitting documents into smaller
pieces before embedding. Chunk size dramatically affects retrieval quality.
Chunks that are too large contain irrelevant content that dilutes the semantic
signal. Chunks that are too small lack the context needed to understand the
chunk's meaning in isolation. The recommended chunk size is 256 to 512 tokens
with 10 to 20 percent overlap between consecutive chunks. Overlap ensures that
sentences at chunk boundaries appear in both adjacent chunks, so retrieval
never misses a key sentence. RecursiveCharacterTextSplitter is the recommended
default because it respects natural language boundaries."""
    },
    {
        "title": "Embeddings and Vector Search",
        "content": """Embeddings are dense numerical vectors that represent the semantic
meaning of text. The all-MiniLM-L6-v2 model produces 384-dimensional vectors.
Vector databases store these embeddings and support approximate nearest neighbor
search, finding the most semantically similar vectors to a query vector. Cosine
similarity is the standard distance metric because it is magnitude-invariant:
a short and a long text about the same topic will have high cosine similarity
even though their vector magnitudes differ. A similarity score above 0.7 typically
indicates highly relevant content, while scores below 0.35 indicate irrelevant content."""
    },
    {
        "title": "RAG Failure Modes",
        "content": """RAG can fail in several ways. Retrieval failures occur when the
wrong chunks are retrieved due to poor embedding quality or bad chunking strategy.
The model may also ignore retrieved context and hallucinate anyway, especially
for long contexts where key information gets buried in the middle. Chunk boundary
failures happen when an answer spans two chunks and neither contains the complete
information. Advanced mitigations include hybrid search combining vector and BM25
keyword search, re-ranking retrieved chunks with a cross-encoder model, and
sentence-window retrieval where individual sentences are indexed but surrounding
context is returned. Evaluating RAG with metrics like context precision, context
recall, and answer faithfulness using RAGAS is strongly recommended."""
    },
]


def ingest_documents(
    docs: List[Dict],
    collection_name: str = "rag_knowledge_base"
) -> chromadb.Collection:
    print("=== INGEST PHASE ===\n")
    client = chromadb.PersistentClient(path="./rag_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    all_ids    = []
    all_metas  = []

    for doc in docs:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:12]
            all_chunks.append(chunk)
            all_ids.append(f"{chunk_id}_{i}")
            all_metas.append({
                "source_title": doc["title"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metas
    )

    print(f"Ingested {len(docs)} documents → {len(all_chunks)} chunks")
    print(f"Stored in collection: '{collection_name}'\n")
    return collection

def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 3,
    min_similarity: float = 0.30
) -> List[Dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    retrieved = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        similarity = 1 / (1 + dist)

        if similarity >= min_similarity:
            retrieved.append({
                "text": doc,
                "source": meta["source_title"],
                "score": round(similarity, 4)
            })

    return retrieved

llm = OllamaLLM(model="llama3.2", temperature=0)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise technical assistant.
Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite which source your answer came from."""),

    ("human", """Context from knowledge base:
{context}

Question: {question}

Answer based on the context above:""")
])

rag_chain = rag_prompt | llm | StrOutputParser()


def answer_question(
    question: str,
    collection: chromadb.Collection
) -> Dict:
    chunks = retrieve(question, collection, top_k=3)

    if not chunks:
        return {
            "answer": "I could not find relevant information in the knowledge base to answer this question.",
            "sources": [],
            "chunks_used": 0
        }

    context = "\n\n".join([
        f"[Source: {c['source']} | Relevance: {c['score']}]\n{c['text']}"
        for c in chunks
    ])

    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": list({c["source"] for c in chunks}),
        # set comprehension: deduplicate source titles
        "chunks_used": len(chunks),
        "retrieved_chunks": chunks
    }

def main():
    collection = ingest_documents(KNOWLEDGE_BASE)

    questions = [
        "What is RAG and why is it useful?",
        "How does the chunking strategy affect RAG quality?",
        "What are the failure modes of RAG and how do you fix them?",
        "How does cosine similarity work for vector search?",
        "What is the difference between the indexing and retrieval phases?",
        "What is the best chunk size to use?",
    ]

    print("=== RAG Q&A SYSTEM ===\n")
    print("=" * 60)

    for question in questions:
        print(f"\n❓ QUESTION: {question}")
        print("-" * 60)

        result = answer_question(question, collection)

        print(f"💬 ANSWER:\n{result['answer']}")
        print(f"\n📚 Sources used ({result['chunks_used']} chunks): {result['sources']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
    