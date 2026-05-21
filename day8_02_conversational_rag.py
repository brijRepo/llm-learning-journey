from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferWindowMemory

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="./day8_chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.6,
    }
)

memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

condense_prompt = PromptTemplate.from_template("""
Given the conversation history below and a follow-up question,
rephrase the follow-up question to be a complete standalone question.
The standalone question should make sense without needing the history.
Do NOT answer the question — only rephrase it.

History:
{chat_history}

Follow-up question: {question}

Standalone question:""")

qa_system_template = """You are a precise LLM engineering tutor.
Answer the question using ONLY the context provided.
If you don't know, say so. Be concise and cite sources.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(qa_system_template),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = OllamaLLM(model="llama3.2", temperature=0)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=condense_prompt,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
    verbose=False
)

def chat(question: str) -> str:
    result = conv_chain.invoke({"question": question})
    sources = {doc.metadata.get("source", "Unknown")
               for doc in result.get("source_documents", [])}
    print(f"{question}")
    print(f"{result['answer']}")
    print(f"Sources: {sources}")
    print()
    return result["answer"]

print("=== CONVERSATIONAL RAG — Multi-Turn Test ===\n")
print("Watch how 'it', 'they', 'that' get resolved using history.\n")
print("=" * 60 + "\n")

chat("What is RAG and how does the pipeline work?")
chat("What are its main failure modes?")
chat("How do you fix the retrieval failure mode specifically?")
chat("Tell me about LangChain's core abstractions.")
chat("Which of those abstractions did we use to build this chat system?")
