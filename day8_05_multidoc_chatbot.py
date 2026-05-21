from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from typing import List, Dict
import os
import shutil

SOURCES: List[Dict] = [
    {
        "source_name": "LLM Fundamentals Guide",
        "content": """
Large Language Models (LLMs) are neural networks trained on vast text corpora
using the next-token prediction objective. The Transformer architecture,
introduced in 'Attention Is All You Need' (2017), is the foundation of all
modern LLMs. Transformers use self-attention to capture relationships between
all tokens simultaneously, unlike RNNs which process sequentially.

Tokenization converts raw text into integer token IDs using algorithms like
Byte-Pair Encoding (BPE) or WordPiece. One token is approximately 0.75 English
words. Context window limits, API pricing, and inference speed are all measured
in tokens. Engineers must actively manage token budgets in production systems.

Temperature controls output randomness during sampling. At temperature 0, the
model always picks the highest-probability token (greedy decoding). Higher
temperatures flatten the distribution, producing more varied and creative
outputs. For factual or structured tasks, temperature 0 to 0.3 is recommended.
        """
    },
    {
        "source_name": "Production RAG Handbook",
        "content": """
Building production RAG systems requires careful attention to five components:
ingestion pipeline, chunking strategy, embedding model selection, vector store
configuration, and retrieval tuning.

Ingestion pipelines should be idempotent — running the same document twice
should not create duplicates. Use content hashing to generate stable document
IDs. Monitor ingestion failures and build retry logic for failed embeddings.

Chunking strategy is the most impactful parameter in RAG. Use RecursiveCharacterTextSplitter
with 256-512 token chunks and 10-20% overlap as a default. For structured
documents like legal contracts or scientific papers, consider semantic chunking
to respect natural topic boundaries. For conversational data, chunk by dialogue turn.

Embedding model selection affects retrieval quality and cost. all-MiniLM-L6-v2
(384 dims) is fast and free locally. text-embedding-ada-002 (1536 dims) is
OpenAI's standard. BAAI/bge-large-en-v1.5 is a strong open-source alternative
at 1024 dims. Always use the same model for ingestion and querying.

Evaluate RAG systems using RAGAS metrics: context precision, context recall,
answer faithfulness, and answer relevance. Set up a golden Q&A test set of
50-100 questions and run evaluations on every pipeline change.
        """
    },
    {
        "source_name": "LangChain Engineering Patterns",
        "content": """
LangChain Expression Language (LCEL) uses the pipe operator to compose
Runnable components. Any class implementing the Runnable interface can be
connected: PromptTemplate | LLM | OutputParser. LCEL chains are lazily
evaluated, support streaming natively, and enable parallel execution with
RunnableParallel.

Agents in LangChain use the ReAct pattern: Thought → Action → Observation
repeated until a final answer is reached. The AgentExecutor handles the loop,
tool dispatch, and error recovery. max_iterations prevents infinite loops.
Tools are Python functions decorated with @tool whose docstrings tell the
agent when to use them.

Memory in LangChain comes in several types. ConversationBufferMemory stores
the full history. ConversationBufferWindowMemory keeps the last K turns.
ConversationSummaryMemory uses an LLM to compress old history into a summary.
ConversationSummaryBufferMemory combines recent verbatim history with a
summary of older turns — best for long conversations.

Callbacks in LangChain enable logging, tracing, and monitoring. Implement
BaseCallbackHandler to intercept events like on_llm_start, on_tool_end,
on_chain_error. Integrate with LangSmith for production-grade tracing and
evaluation dashboards.
        """
    },
    {
        "source_name": "AI Security & Safety Notes",
        "content": """
Prompt injection attacks attempt to override developer instructions by embedding
adversarial content in user-controlled inputs. Indirect prompt injection occurs
when the attack is embedded in external content the model processes, such as a
web page or uploaded document. Mitigations: strict role separation, input
sanitization, output validation, and least-privilege tool access.

Jailbreaking attempts to bypass safety guidelines using techniques like role
play ('pretend you are an AI without restrictions'), hypothetical framing
('in a fictional story where...'), and token manipulation. Modern frontier
models have improved resistance through RLHF and Constitutional AI training.

Data privacy in LLM apps requires careful handling of PII. Never log raw
user messages containing sensitive data. Implement data retention policies.
Use local models for sensitive workloads to avoid sending data to external APIs.
Consider differential privacy for fine-tuning on sensitive datasets.

Model output validation is underutilized in production. Validate structured
outputs against schemas. Detect and filter hallucinated URLs, citations, or
code before displaying to users. Implement human-in-the-loop review for
high-stakes decisions.
        """
    },
]

def build_vectorstore(sources: List[Dict]) -> Chroma:
    DB_PATH = "./day8_multidoc_db"

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=90,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_texts:     List[str]  = []
    all_metadatas: List[Dict] = []

    for source in sources:
        chunks = splitter.split_text(source["content"].strip())
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "source": source["source_name"],
                "chunk_index": i,
            })

    vectorstore = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=DB_PATH
    )

    print(f"Vectorstore built: {len(all_texts)} chunks from {len(sources)} sources")
    return vectorstore


def build_chatbot(vectorstore: Chroma) -> ConversationalRetrievalChain:
    llm = OllamaLLM(model="llama3.2", temperature=0)

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.65}
    )

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=OllamaLLM(model="llama3.2", temperature=0.3)
    )

    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    condense_prompt = PromptTemplate.from_template("""
Given the chat history and a follow-up question, rephrase the
follow-up into a complete standalone question. Only rephrase — do not answer.

History: {chat_history}
Follow-up: {question}
Standalone question:""")
    
    system_template = """You are an expert LLM engineering tutor with access
to a multi-source knowledge base. Answer using ONLY the provided context.
Always cite which source your answer comes from.
If the context doesn't contain the answer, say:
"I don't have information about that in my knowledge base."

Context from knowledge base:
{context}"""

    answer_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=multi_retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain

def run_chatbot():
    print("\n" + "═" * 60)
    print("MULTI-DOC RAG CHATBOT  ")
    print("  Sources loaded:")
    for s in SOURCES:
        print(f"    • {s['source_name']}")
    print("\n  Commands: 'exit' | 'clear' | 'sources'")
    print("═" * 60 + "\n")

    print("Initializing vectorstore and chain...")
    vectorstore = build_vectorstore(SOURCES)
    chain = build_chatbot(vectorstore)
    print("Ready!\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\nChatbot: Goodbye! Keep building. 🚀\n")
            break

        if user_input.lower() == "clear":
            chain = build_chatbot(vectorstore)
            print("Chatbot: Memory cleared. Fresh conversation!\n")
            continue

        if user_input.lower() == "sources":
            print("\nAvailable sources in knowledge base:")
            for s in SOURCES:
                print(f"  • {s['source_name']}")
            print()
            continue

        result = chain.invoke({"question": user_input})

        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        sources_used = list({
            doc.metadata.get("source", "Unknown")
            for doc in source_docs
        })

        print(f"\nChatbot: {answer}")
        if sources_used:
            print(f"[{len(source_docs)} chunks from: {', '.join(sources_used)}]")
        print()


if __name__ == "__main__":
    run_chatbot()
    