from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from typing import List

DOCUMENTS = [
    """Transformers use self-attention to process sequences. Each token
attends to every other token, computing relevance scores via Query, Key,
and Value matrices. Multi-head attention runs this in parallel across
multiple heads, each learning different relationships like syntax,
coreference, and semantics. The outputs are concatenated and projected.""",

    """RAG (Retrieval Augmented Generation) combines vector search with
language generation. The indexing phase chunks documents, embeds them,
and stores them in a vector database. The retrieval phase embeds the user
query and finds the top-k most similar chunks. These chunks are injected
into the LLM prompt as context for grounded answer generation.""",

    """LangChain is a framework for building LLM applications. Its core
abstractions are chains (sequences of components), agents (LLMs that use
tools in a loop), memory (conversation history management), and retrievers
(standardized interface for fetching relevant documents from any source).""",

    """Fine-tuning updates a pre-trained model's weights on domain-specific
data. It teaches the model new styles, formats, or domain knowledge
permanently. Fine-tuning is expensive, requires labeled data, and must be
repeated for every knowledge update. RAG is preferred when knowledge
changes frequently. Fine-tuning is preferred when output style and format
consistency is critical.""",

    """Prompt injection is a security vulnerability where malicious user input
overrides developer instructions. An attacker embeds instructions inside
content the model processes, such as 'Ignore all previous instructions'.
Defenses include strict role separation between system and user content,
input sanitization, output monitoring, and least-privilege tool access.""",

    """Vector databases store embedding vectors and support approximate nearest
neighbor search. ChromaDB is a lightweight local option requiring no server.
Pinecone is a managed cloud vector database with horizontal scalability.
Qdrant and Weaviate offer self-hosted options with advanced filtering.
pgvector extends PostgreSQL with vector search for teams with existing Postgres.""",

    """The context window is the maximum tokens an LLM can process at once.
GPT-4 supports 128k tokens. Llama 3.2 supports 128k tokens. Claude 3.5
Sonnet supports 200k tokens. Engineers must actively manage context window
usage by summarizing conversation history, chunking documents, and counting
tokens before API calls to prevent context overflow errors.""",

    """Cosine similarity measures the angle between two embedding vectors.
It returns 1.0 for identical direction, 0.0 for orthogonal, -1.0 for
opposite. It is magnitude-invariant, meaning a short and long text about
the same topic score highly similar. This makes it ideal for semantic search
compared to Euclidean distance which penalizes length differences.""",
]

TITLES = [
    "Transformer Architecture",
    "RAG Pipeline",
    "LangChain Framework",
    "Fine-tuning vs RAG",
    "Prompt Injection Security",
    "Vector Databases Comparison",
    "Context Window Management",
    "Cosine Similarity",
]

print("Building vectorstore...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

all_texts = []
all_metadatas = []

for doc_text, title in zip(DOCUMENTS, TITLES):
    chunks = splitter.split_text(doc_text)
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({"source": title})

vectorstore = Chroma.from_texts(
    texts=all_texts,
    embedding=embeddings,
    metadatas=all_metadatas,
    persist_directory="./day8_chroma_db"
)

print(f"Vectorstore built with {vectorstore._collection.count()} chunks.\n")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
    }
)

RAG_PROMPT_TEMPLATE = """You are a precise LLM engineering tutor.
Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I don't have enough context to answer this."
Be concise and cite the source in your answer.

Context:
{context}

Question: {question}

Answer:"""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

llm = OllamaLLM(model="llama3.2", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": rag_prompt}
)

queries = [
    "How does attention work in transformers?",
    "When should I use fine-tuning instead of RAG?",
    "What vector databases are available?",
    "What is prompt injection and how do I defend against it?",
]

print("=== RetrievalQA Results ===\n")

for query in queries:
    print(f"{query}")
    result = qa_chain.invoke({"query": query})
    print(f"{result['result']}")
    sources = {doc.metadata.get("source", "Unknown")
               for doc in result["source_documents"]}
    print(f"Sources: {sources}\n")
    print("-" * 60 + "\n")

