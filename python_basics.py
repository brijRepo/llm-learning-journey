import json

# Variables (containers that hold data)
name = "Brijesh"
age = 33
salary = 2500.50
is_learning = True

print(name)
print(age)
print(f"My name is {name} and I am {age} years old")

print(type(name))
print(type(age))



# List: ordered, mutable (can change)
fruits = ["banana", "orange", "grape"]
numbers = [1, 2, 3]
# Access by index (starts at 0)
print(fruits[0])
# Add items
fruits.append("kiwi")
print(fruits)
# Length
print(len(fruits))
# Loop through list
for fruit in fruits:
    print(f"I like {fruit}")



# Dictionary: like a real dictionary (word → definition)
person = {
    "name": "Brijesh",
    "age": 33,
    "job": "SE"
}
# Access by key
print(person["name"])
# Add new key
person["salary"] = 2500
print(person)
# Loop through dictionary
for key, value in person.items():
    print(f"{key}: {value}")


# Function: reusable block of code
def greet(name):
    return f"Hello {name}!"
# Call the function
message = greet("Brijesh")
print(message)
# Function with multiple parameters
def calculate_cost(query_value, query_count):
    total = query_count * query_value
    return total

monthly_cost = calculate_cost(250, 40)
print(f"Total value is {monthly_cost}")


# Control Flow
def check_budget(cost, budget):
    if cost > budget:
        return "Over-budget"
    elif cost == budget:
        return "Exactly at budget"
    else:
        return "Over-budget"

result = check_budget(50, 55)
print(result)



# For loop (when you know how many times)
for i in range(3):
    print(f"Iteration {i}")
# While loop (when you don't know how many times)
count = 4
while(count < 22):
    print(f"Count {count}")
    count += 2


#Function that takes a list of number and then returns the sum
def sum(ls):
    sum = 0
    for number in ls:
        sum = sum + number
    return sum

list_of_number = [2, 4, 7]
total = sum(list_of_number)
print(f"The sum is {total}")



# Write to file
with open("text.txt", "w") as file:
    file.write("This is first line\n")
    file.write("This is second line\n")
# Read from file
with open("text.txt", "r") as file:
    content = file.read()
    print(content)
# Read line by line
with open("text.txt", "r") as file:
    for line in file:
        print(line.strip())


#JSON(Critical for LLM work)
# Python dictionary
data = {
  "name": "Claude",
  "model": "Sonnet",
  "token": 2000
}
# Convert to JSON string
data_dump = json.dumps(data)
print(data_dump)
# Write JSON to file
with open("data.json", "w") as file:
    json.dump(data, file, indent=2)
# Read JSON from file
with open("data.json", "r") as file:
    loaded_json = json.load(file)
    print(loaded_json["model"])


# With error handling
try:
    with open("test.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("The file doesn't exist. Creating it!")
    with open("test.txt", "w") as file:
        file.write("This is the content of the file")
except Exception as e:
    print(f"Exception {e} occured")



