
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
import datetime

llm = OllamaLLM(model="llama3.2", temperature=0.5)
parser = StrOutputParser()

@tool
def get_current_date() -> str:
    """Returns today's date. Use when user asks about current date or day."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y")

@tool
def explain_concept(concept: str) -> str:
    """
    Explains a core LLM or AI engineering concept briefly.
    Use when the user asks to define or explain a technical term.

    Args:
        concept: The technical concept to explain
    """
    definitions = {
        "rag": "RAG retrieves relevant docs at query time and injects them as prompt context.",
        "temperature": "Temperature controls randomness. 0=deterministic, 1+=creative.",
        "embeddings": "Embeddings are vectors representing text semantics. Similar text = close vectors.",
        "langchain": "LangChain is a framework for composing LLM applications using chains, agents, and memory.",
        "agent": "An LLM agent uses tools in a Thought→Action→Observation loop to complete tasks.",
        "chain": "A chain is a sequence of LangChain components (prompt | llm | parser) connected via LCEL.",
    }
    for k, v in definitions.items():
        if k in concept.lower():
            return v
    return f"No definition found for '{concept}'. Available: {list(definitions.keys())}"

tool_registry = {
    "get_current_date": get_current_date,
    "explain_concept": explain_concept
}

tools_description = """
You have access to these tools. Use them when appropriate:
1. get_current_date() - Call when user asks about today's date
2. explain_concept(concept) - Call when user asks to define a technical term

To use a tool, respond EXACTLY in this format (nothing else on that line):
TOOL: tool_name | argument

Example: TOOL: explain_concept | embeddings
If no tool is needed, respond normally.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are Aria, a knowledgeable LLM engineering tutor.
You help engineers learn to build production-grade LLM applications.
You are concise, practical, and use concrete examples.

{tools_description}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | parser

def maybe_execute_tool(response: str) -> str:
    if not response.strip().startswith("TOOL:"):
        return response
    try:
        parts = response.replace("TOOL:", "").strip().split("|")
        tool_name = parts[0].strip()
        tool_arg = parts[1].strip() if len(parts) > 1 else ""
        if tool_name in tool_registry:
            tool_fn = tool_registry[tool_name]
            if tool_arg:
                tool_result = tool_fn.invoke(tool_arg)
            else:
                tool_result = tool_fn.invoke({})

            return f"[Tool: {tool_name}] → {tool_result}"
        else:
            return f"[Unknown tool: {tool_name}]"
    except Exception as e:
        return f"[Tool error: {e}]"
    
def run_chat_app():
    print("\n" + "="*55)
    print("  🤖  ARIA — LLM Engineering Tutor  ")
    print("  Type 'exit' to quit | 'clear' to reset memory")
    print("="*55 + "\n")

    history = []
    WINDOW = 10
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Aria: Goodbye! Keep building. 🚀")
            break
        if user_input.lower() == "clear":
            history = []
            print("Aria: Memory cleared. Fresh start!\n")
            continue

        response = chain.invoke({
            "history": history[-WINDOW:],
            # Slice last WINDOW messages for context management
            "input": user_input
        })
        final_response = maybe_execute_tool(response)

        if final_response.startswith("[Tool:"):
            followup_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are Aria, an LLM engineering tutor. Be concise."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
                ("assistant", "{tool_result}"),
                ("human", "Now answer the user's original question using the tool result above.")
            ])
            followup_chain = followup_prompt | llm | parser
            final_response = followup_chain.invoke({
                "history": history[-WINDOW:],
                "input": user_input,
                "tool_result": final_response
            })
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=final_response))

        print(f"\nAria: {final_response}\n")

if __name__ == "__main__":
    run_chat_app()
    