from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.2", temperature=0.7)
parser = StrOutputParser()


prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor specializing in LLM engineering."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chain = prompt_with_history | llm | parser

chat_history = []

def chat(user_input: str) -> str:
    response = chain.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    return response

print("=== MANUAL MEMORY CHAT ===\n")

r1 = chat("What is RAG in one sentence?")
print(f"User: What is RAG in one sentence?")
print(f"AI: {r1}\n")

r2 = chat("What are its main failure modes?")
print(f"User: What are its main failure modes?")
print(f"AI: {r2}\n")

r3 = chat("How does the second failure mode get fixed?")
print(f"User: How does the second failure mode get fixed?")
print(f"AI: {r3}\n")

print(f"--- History has {len(chat_history)} messages ---")

def chat_windowed(user_input: str, history: list, window: int = 6) -> tuple:
    windowed_history = history[-window:]

    response = chain.invoke({
        "chat_history": windowed_history,
        "user_input": user_input
    })

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))

    return response, history

print("\n=== WINDOWED MEMORY CHAT ===\n")
windowed_history = []

for user_msg in [
    "My name is Brijesh.",
    "I am learning LLM engineering.",
    "What is my name?",
    "What am I learning?",
]:
    response, windowed_history = chat_windowed(user_msg, windowed_history, window=6)
    print(f"User: {user_msg}")
    print(f"AI:   {response}\n")
    