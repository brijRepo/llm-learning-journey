from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(
    model= "llama3.2",
    temperature= 0.7,
)

prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template="""You are a technical educator.
    Explain the concept of {topic} to a {audience}.
    Keep your explanation under 100 words and use an analogy."""
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "topic": " vector embeddings",
    "audience": "junior developer"
})

print("=== RESULT ===")
print(result)

result2 = chain.invoke({
    "topic": "RAG (Retrieval Augmented Generation)",
    "audience": "non-technical product manager"
})

print("\n=== RESULT 2 ===")
print(result2)

formatted = prompt.format(
    topic="attention mechanism",
    audience="high school student"
)

print("\n=== FORMATTED PROMPT (no LLM call) ===")
print(formatted)
