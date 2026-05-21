from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="./day8_chroma_db",
    embedding_function=embeddings
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = OllamaLLM(model="llama3.2", temperature=0)

llm_extractor = LLMChainExtractor.from_llm(llm)
llm_compression_retriever = ContextualCompressionRetriever(
    base_compressor=llm_extractor,
    base_retriever=base_retriever,
)

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.50,
)
embedding_compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever,
)

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=". "
)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        splitter,
        embeddings_filter,
        llm_extractor,
    ]
)
pipeline_compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever,
)

def show_retrieved_content(retriever, query: str, label: str):
    """Retrieve docs and show the (compressed) content returned."""
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    docs = retriever.get_relevant_documents(query)
    print(f"Chunks returned: {len(docs)}")
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown")
        print(f"\n  Chunk {i+1} [{src}]:")
        print(f"  {doc.page_content}")
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"\n  Total chars in context: {total_chars}")


query = "How should I choose between fine-tuning and RAG?"

print("=" * 55)
print(f"QUERY: {query}")
print("=" * 55)

show_retrieved_content(base_retriever,                  query, "BASELINE (no compression)")
show_retrieved_content(embedding_compression_retriever, query, "EMBEDDINGS FILTER")
show_retrieved_content(llm_compression_retriever,       query, "LLM EXTRACTOR")
show_retrieved_content(pipeline_compression_retriever,  query, "PIPELINE (split→filter→extract)")

RAG_PROMPT = PromptTemplate(
    template="""Answer concisely using ONLY the context.

Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"]
)
compressed_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=embedding_compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": RAG_PROMPT}
)

print("\n\n=== COMPRESSED QA CHAIN RESULT ===\n")
result = compressed_qa.invoke({"query": query})
print(f"Answer: {result['result']}")
