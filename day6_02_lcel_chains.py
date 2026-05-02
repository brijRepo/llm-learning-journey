from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = OllamaLLM(model="llama3.2", temperature=0.3)
parser = StrOutputParser()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior Python engineer. Be concise and precise."),
    ("human", "Explain what {concept} does in Python with a code example.")
])

explain_chain = chat_prompt | llm | parser

output = explain_chain.invoke({"concept": "list comprehension"})
print("=== Python Explainer ===")
print(output)

step1_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer. Be clear and educational."),
    ("human", "Explain {topic} in 3 bullet points for a beginner.")
])

step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a quiz creator."),
    ("human", """Based on this explanation:
{explanation}

Create 2 multiple-choice quiz questions to test understanding.
Format: Q: ... A) ... B) ... C) ... D) ... Answer: ...""")
])

step1_chain = step1_prompt | llm | parser

step2_chain = step2_prompt | llm | parser

full_chain = (
    step1_chain
    | (lambda explanation: {"explanation": explanation})
    | step2_chain
)

print("\n=== Multi-Step: Explain → Quiz ===")
quiz_output = full_chain.invoke({"topic": "transformer attention mechanism"})
print(quiz_output)

from langchain_core.runnables import RunnableParallel


parallel_chain = RunnableParallel(
    original_topic=RunnablePassthrough(),
    explanation=step1_chain
)

result = parallel_chain.invoke({"topic": "cosine similarity"})
print("\n=== Parallel (Original + Explanation) ===")
print("Topic received:", result["original_topic"])
print("Explanation:", result["explanation"])