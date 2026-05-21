from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import logging

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="./day8_chroma_db",
    embedding_function=embeddings
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

logging.basicConfig(level=logging.INFO)
mq_logger = logging.getLogger("langchain.retrievers.multi_query")
mq_logger.setLevel(logging.INFO)

llm = OllamaLLM(model="llama3.2", temperature=0.3)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)


def compare_retrievers(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    single_docs = base_retriever.get_relevant_documents(query)

    print(f"\nSINGLE-QUERY: {len(single_docs)} chunks retrieved")
    for i, doc in enumerate(single_docs):
        src = doc.metadata.get("source", "Unknown")
        print(f"  Chunk {i+1} [{src}]: {doc.page_content[:80]}...")

    print(f"\nMULTI-QUERY: (watch generated queries in logs)")
    multi_docs = multi_query_retriever.get_relevant_documents(query)

    print(f"\n  Total unique chunks: {len(multi_docs)}")
    for i, doc in enumerate(multi_docs):
        src = doc.metadata.get("source", "Unknown")
        print(f"  Chunk {i+1} [{src}]: {doc.page_content[:80]}...")

    additional = len(multi_docs) - len(single_docs)
    print(f"\nMulti-query found {max(0, additional)} additional unique chunks")

compare_retrievers("compare embedding distance metrics")
compare_retrievers("how to handle running out of model context")

RAG_PROMPT = PromptTemplate(
    template="""Answer using ONLY the context below.
Cite your source. If context is insufficient, say so.

Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

mq_qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="llama3.2", temperature=0),
    chain_type="stuff",
    retriever=multi_query_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": RAG_PROMPT}
)

print("\n\n=== MULTI-QUERY QA CHAIN ===\n")
result = mq_qa_chain.invoke({
    "query": "What is the difference between using a cloud API and local inference for LLMs?"
})
print(f"Answer: {result['result']}")
sources = {d.metadata.get("source") for d in result["source_documents"]}
print(f"Sources: {sources}")
