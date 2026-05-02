from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import datetime
import json

# @tool
# def get_current_datetime() -> str:
#     """
#     Returns the current date and time.
#     Use this tool whenever the user asks about the current time,
#     today's date, or what day it is.
#     """
#     now = datetime.datetime.now()
#     return now.strftime("%A, %B %d, %Y at %H:%M:%S")
@tool
def get_current_datetime(dummy: str = "") -> str:
    """
    Returns the current date and time.
    Use this tool whenever the user asks about the current time,
    today's date, or what day it is.
    """
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %H:%M:%S")

@tool
def calculate_days_until(target_date: str) -> str:
    """
    Calculates how many days until a given target date.
    Input must be a date string in YYYY-MM-DD format.
    Use this when the user wants to know how many days until
    a specific date or event.

    Args:
        target_date: Date in YYYY-MM-DD format (e.g., '2025-12-31')
    """

    try:
        target_date = target_date.strip().replace("'", "").replace('"', "")
        target = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        today = datetime.datetime.now()
        delta = target - today

        if delta.days < 0:
            return f"That date was {abs(delta.days)} days ago."
        elif delta.days == 0:
            return "That date is today!"
        else:
            return f"{delta.days} days until {target_date}."
    except ValueError as e:
        return f"Invalid date format. Use YYYY-MM-DD. Error: {e}"
    
@tool
def lookup_llm_concept(concept: str) -> str:
    """
    Looks up a definition or explanation of a core LLM concept.
    Use this when the user asks about technical terms like
    'RAG', 'fine-tuning', 'embeddings', 'temperature', 'tokens', etc.

    Args:
        concept: The LLM/AI concept to look up
    """
    knowledge_base = {
        "rag": "RAG (Retrieval-Augmented Generation) retrieves relevant documents at query time and injects them into the LLM prompt as context, grounding the response in real data.",
        "temperature": "Temperature controls output randomness. 0 = deterministic (always picks highest probability token). 1.0 = very random. Use low values for factual tasks.",
        "embeddings": "Embeddings are dense vector representations of text where semantic similarity = spatial proximity. Similar meanings → vectors that point in similar directions.",
        "fine-tuning": "Fine-tuning updates model weights on domain-specific data to teach new behavior or style. Expensive but permanent — unlike RAG which is dynamic.",
        "tokens": "Tokens are subword units that LLMs process. ~0.75 words per token in English. Context limits, pricing, and speed are all measured in tokens.",
        "attention": "Attention allows each token to 'look at' all other tokens when computing its representation. It's the core mechanism that made Transformers so powerful.",
    }

    key = concept.lower().strip()

    for k, v in knowledge_base.items():
        if k in key or key in k:
            return v
        
    return f"Concept '{concept}' not found in knowledge base. Try: {list(knowledge_base.keys())}"

tools = [get_current_datetime, calculate_days_until, lookup_llm_concept]
llm = OllamaLLM(model="llama3.2", temperature=0)
# react_prompt = hub.pull("hwchase17/react")
from langchain_core.prompts import PromptTemplate

react_prompt = PromptTemplate.from_template("""
You are an AI agent that MUST follow this format strictly:

You have access to the following tools:
{tools}

Tool names:
{tool_names}

Use this format:

Thought: what you think
Action: one of [{tool_names}]
Action Input: input to the tool

OR

Final Answer: your final answer

Begin!

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, react_prompt)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     max_iterations=5,
#     handle_parsing_errors=True 
# )
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=2,  # reduce looping
    handle_parsing_errors=True,
    early_stopping_method="force"
)

print("=== AGENT WITH TOOLS ===\n")

queries = [
    "What is today's date and time?",
    "How many days until 2025-12-31?",
    "Can you explain what RAG means?",
    "What is the current time and also explain temperature in LLMs?"
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"USER: {query}")
    print(f"{'='*50}")
    result = agent_executor.invoke({"input": query})
    print(f"\nFINAL ANSWER: {result['output']}")
