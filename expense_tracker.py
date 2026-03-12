import json
from datetime import datetime

# Initialize expenses list
expenses = []

def add_expense(amount, category, description):
    """Add a new expense to the list."""
    expense = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "amount": amount,
        "category": category,
        "description": description
    }

    expenses.append(expense)
    print(f"Added: ${amount} for {category}")


def save_expenses():
    """Save expenses to JSON file."""
    with open("expenses.json", "w") as file:
        json.dump(expenses, file, indent=2)

    print("Expenses saved!")


def load_expenses():
    """Load expenses from JSON file."""
    global expenses

    try:
        with open("expenses.json", "r") as file:
            expenses = json.load(file)

        print(f"Loaded {len(expenses)} expenses")

    except FileNotFoundError:
        print("No existing expenses file. Starting fresh.")


def show_total():
    """Calculate and show total expenses."""
    total = sum(expense["amount"] for expense in expenses)
    print(f"Total expenses: ${total:.2f}")


# def show_by_category(category):
#     for expense in expenses:
#         print(f'The category {expense["category"]} has ${expense["amount"]}')

def show_by_category(category):
    """Show expenses for the given category."""

    for expense in expenses:
        if expense["category"].lower() == category.lower():
            print(f'{expense["date"]} - ${expense["amount"]} - {expense["description"]}')

def delete_expense(index):
    """Delete an expense using its index."""

    if 0 <= index < len(expenses):
        removed = expenses.pop(index)
        print(f'Removed: {removed["date"]} - ${removed["amount"]} - {removed["description"]}')
    else:
        print("Invalid index")


# Main program
if __name__ == "__main__":
    load_expenses()

    # Add some expenses
    # add_expense(50.00, "Food", "Groceries")
    # add_expense(120.00, "Tech", "OpenAI API credits")
    # add_expense(30.00, "Food", "Lunch")

    show_total()
    save_expenses()

    show_by_category("Food")
    delete_expense(2)

    load_expenses()