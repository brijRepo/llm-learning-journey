import ollama
import json
import re
from datetime import datetime

def calculator(operation, a, b):
    operations = {
        "add": a+b,
        "subtract": a-b,
        "multiply": a*b,
        "divide": a/b if b != 0 else "Error: Division by zero"
    }
    return operations.get(operation, "Error: Unknown operation")

def get_current_time():
    return datetime.now.strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city):
    weather_db = {
        "new york": "Sunny, 72°F",
        "london": "Rainy, 58°F",
        "tokyo": "Cloudy, 65°F",
        "paris": "Partly cloudy, 68°F",
        "mumbai": "Hot and humid, 88°F",
        "pune": "Pleasant, 75°F"
    }

    city_lower = city.lower()
    return weather_db.get(city_lower, f"Weather data not available for {city}")

def search_web(query):
    return f"Top results for '{query}': [Simulated search results would appear here]"

TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Performs arithmetic operations (add, subtract, multiply, divide)",
        "parameters": {
            "operation": "string (add|subtract|multiply|divide)",
            "a": "number",
            "b": "number"
        },
        "example": '{"tool": "calculator", "operation": "add", "a": 5, "b": 3}'
    },
    "get_current_time": {
        "function": get_current_time,
        "description": "Returns current date and time",
        "parameters": {},
        "example": '{"tool": "get_current_time"}'
    },
    "get_weather": {
        "function": get_weather,
        "description": "Gets weather information for a city",
        "parameters": {
            "city": "string (city name)"
        },
        "example": '{"tool": "get_weather", "city": "London"}'
    },
    "search_web": {
        "function": search_web,
        "description": "Searches the web for information",
        "parameters": {
            "query": "string (search query)"
        },
        "example": '{"tool": "search_web", "query": "Python tutorial"}'
    }
}

def create_system_prompt():
    tools_description = "You have access to these tools:\n\n"
    
    for tool_name, tool_info in TOOLS.items():
        tools_description += f"**{tool_name}**\n"
        tools_description += f"- Description: {tool_info['description']}\n"
        tools_description += f"- Parameters: {tool_info['parameters']}\n"
        tools_description += f"- Example: {tool_info['example']}\n\n"
    
    system_prompt = f"""{tools_description}

    To use a tool, respond with:
    TOOL_CALL: {{"tool": "tool_name", "param1": value1, ...}}

    After using a tool, I will give you the result, then you can answer the user.

    If you don't need tools, just answer normally."""
    
    return system_prompt

def execute_tool(tool_call):
    tool_name = tool_call.get('tool')
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    
    tool_func = TOOLS[tool_name]['function']
    params = {k: v for k, v in tool_call.items() if k != 'tool'}

    try:
        # Execute function with parameters
        if params:
            result = tool_func(**params)
        else:
            result = tool_func()
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def chat_with_tools(user_message, max_iterations = 3):
    system_prompt = create_system_prompt()
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_message}
    ]

    for iteration in range(max_iterations):
        # Get AI response
        response = ollama.chat(
            model='llama3.2:3b',
            messages=messages
        )
        
        ai_response = response['message']['content']

        if "TOOL_CALL:" in ai_response:
            print(f"  [Iteration {iteration + 1}: AI calling tool...]")

            match = re.search(r'TOOL_CALL:\s*(\{.*?\})', ai_response, re.DOTALL)

            if match:
                try:
                    tool_call = json.loads(match.group(1))
                    print(f"  [Tool: {tool_call.get('tool', 'unknown')}]")
                    
                    result = execute_tool(tool_call)
                    print(f"  [Result: {result}]")
                    
                    messages.append({'role': 'assistant', 'content': ai_response})
                    messages.append({'role': 'user', 'content': f"Tool result: {result}"})
                    
                except json.JSONDecodeError as e:
                    print(f"  [Error parsing tool call: {e}]")
                    return ai_response
            else:
                return ai_response
        else:
            return ai_response
        
    return "Error: Max tool iterations reached"


def main():
    print("Multi-Tool Agent Demo")
    print("=" * 70)
    print("I have access to: calculator, time, weather, web search\n")
    
    test_queries = [
        "What time is it?",
        "What's 456 times 789?",
        "What's the weather in London?",
        "Search for Python tutorials",
        "What's the weather in Tokyo and what time is it there?",  # Multiple tools!
    ]
    
    for query in test_queries:
        print(f"\nYou: {query}")
        print("-" * 70)
        answer = chat_with_tools(query)
        print(f"\nAI: {answer}")
        print("=" * 70)


if __name__ == "__main__":
    main()
