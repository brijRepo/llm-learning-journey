import ollama
import json
import re
import os
from datetime import datetime

TASKS_FILE = "assistant_tasks.json"
NOTES_FILE = "assistant_notes.json"


def load_json_file(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {"items": []}

def save_json_file(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def add_task(task, priority="medium"):
    data = load_json_file(TASKS_FILE)
    new_task = {
        "id": len(data["items"]) + 1,
        "task": task,
        "priority": priority,
        "created": datetime.now().isoformat(),
        "completed": False
    }
    data["items"].append(new_task)
    save_json_file(TASKS_FILE, data)
    return f"Added task #{new_task['id']}: {task} (priority: {priority})" 

def list_tasks():
    data = load_json_file(TASKS_FILE)
    active_tasks = [t for t in data["items"] if not t["completed"]]
    
    if not active_tasks:
        return "No active tasks!"
    
    result = f"Active tasks ({len(active_tasks)}):\n"
    for task in active_tasks:
        result += f"  #{task['id']} [{task['priority']}] {task['task']}\n"
    
    return result

def complete_task(task_id):
    data = load_json_file(TASKS_FILE)

    for task in data["items"]:
        if task["id"] == task_id:
            task["completed"] = True
            task["completed_at"] = datetime.now().isoformat()
            save_json_file(TASKS_FILE, data)
            return f"Completed task #{task_id}: {task['task']}"
    
    return f"Task #{task_id} not found"

def add_note(title, content):
    data = load_json_file(NOTES_FILE)
    
    new_note = {
        "id": len(data["items"]) + 1,
        "title": title,
        "content": content,
        "created": datetime.now().isoformat()
    }
    
    data["items"].append(new_note)
    save_json_file(NOTES_FILE, data)
    
    return f"Saved note #{new_note['id']}: {title}"

def search_notes(query):
    data = load_json_file(NOTES_FILE)
    query_lower = query.lower()
    
    matches = [
        note for note in data["items"]
        if query_lower in note["title"].lower() or query_lower in note["content"].lower()
    ]
    
    if not matches:
        return f"No notes found for '{query}'"
    
    result = f"Found {len(matches)} note(s):\n"
    for note in matches:
        result += f"  #{note['id']}: {note['title']}\n"
        result += f"    {note['content'][:100]}...\n"
    
    return result

def search_files(pattern):
    import glob
    
    files = glob.glob(f"*{pattern}*")
    
    if not files:
        return f"No files found matching '{pattern}'"
    
    return f"Found {len(files)} file(s): {', '.join(files)}"

def get_current_time():
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


TOOLS = {
    "add_task": {
        "function": add_task,
        "description": "Add a task to the todo list",
        "parameters": {
            "task": "string (task description)",
            "priority": "string (high|medium|low, default: medium)"
        },
        "example": '{"tool": "add_task", "task": "Finish Day 4", "priority": "high"}'
    },
    "list_tasks": {
        "function": list_tasks,
        "description": "Show all active tasks",
        "parameters": {},
        "example": '{"tool": "list_tasks"}'
    },
    "complete_task": {
        "function": complete_task,
        "description": "Mark a task as complete",
        "parameters": {
            "task_id": "integer (task ID number)"
        },
        "example": '{"tool": "complete_task", "task_id": 1}'
    },
    "add_note": {
        "function": add_note,
        "description": "Save a note",
        "parameters": {
            "title": "string (note title)",
            "content": "string (note content)"
        },
        "example": '{"tool": "add_note", "title": "Meeting Notes", "content": "Discussed..."}'
    },
    "search_notes": {
        "function": search_notes,
        "description": "Search notes by keyword",
        "parameters": {
            "query": "string (search term)"
        },
        "example": '{"tool": "search_notes", "query": "meeting"}'
    },
    "search_files": {
        "function": search_files,
        "description": "Search for files in current directory",
        "parameters": {
            "pattern": "string (filename pattern)"
        },
        "example": '{"tool": "search_files", "pattern": "day"}'
    },
    "get_current_time": {
        "function": get_current_time,
        "description": "Get current date and time",
        "parameters": {},
        "example": '{"tool": "get_current_time"}'
    }
}


def create_system_prompt():
    tools_desc = "You are a helpful personal assistant with access to these tools:\n\n"
    
    for tool_name, tool_info in TOOLS.items():
        tools_desc += f"**{tool_name}**\n"
        tools_desc += f"  {tool_info['description']}\n"
        tools_desc += f"  Parameters: {tool_info['parameters']}\n"
        tools_desc += f"  Example: {tool_info['example']}\n\n"
    
    tools_desc += """
    To use a tool, respond with:
    TOOL_CALL: {"tool": "tool_name", "param1": value1, ...}

    You can use multiple tools if needed. After getting tool results, answer the user's question naturally."""

    return tools_desc

def execute_tool(tool_call):
    tool_name = tool_call.get('tool')
    
    if tool_name not in TOOLS:
        return f"Error: Unknown tool '{tool_name}'"
    
    tool_func = TOOLS[tool_name]['function']
    params = {k: v for k, v in tool_call.items() if k != 'tool'}
    
    try:
        if params:
            result = tool_func(**params)
        else:
            result = tool_func()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
def chat(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    system_prompt = create_system_prompt()

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': user_message})
    
    max_iterations = 5

    for iteration in range(max_iterations):
        response = ollama.chat(
            model='llama3.2:3b',
            messages=messages
        )
        ai_response = response['message']['content']

        if "TOOL_CALL:" in ai_response:
            print(f"Using tool...")

            match = re.search(r'TOOL_CALL:\s*(\{.*?\})', ai_response, re.DOTALL)
            
            if match:
                try:
                    tool_call = json.loads(match.group(1))
                    tool_name = tool_call.get('tool', 'unknown')
                    print(f"{tool_name}")
                    
                    result = execute_tool(tool_call)
                    print(f"{result[:60]}...")
                    
                    messages.append({'role': 'assistant', 'content': ai_response})
                    messages.append({'role': 'user', 'content': f"Tool result: {result}"})
                    
                except json.JSONDecodeError as e:
                    return ai_response, conversation_history + [
                        {'role': 'user', 'content': user_message},
                        {'role': 'assistant', 'content': ai_response}
                    ]
        else:
            # Done, return response
            conversation_history.append({'role': 'user', 'content': user_message})
            conversation_history.append({'role': 'assistant', 'content': ai_response})
            return ai_response, conversation_history
    
    return "Error: Max iterations reached", conversation_history

def main():
    print("Personal Assistant")
    print("=" * 70)
    print("I can help you with:")
    print("Tasks (add, list, complete)")
    print("Notes (save, search)")
    print("Files (search)")
    print("Time")
    print("\nType 'quit' to exit\n")
    
    conversation = []
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue
        
        print()
        response, conversation = chat(user_input, conversation)
        print(f"\nAssistant: {response}\n")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
