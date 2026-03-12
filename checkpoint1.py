# Write a program that creates a JSON file with your info (name, target salary, current dayof learning).
# Then read it back and print.

import json

data1 = {
    "name": "Brijesh",
    "target_salary": 25000,
    "current_day_of_learning": 2
}

with open("data1.json", "w") as file:
    json.dump(data1, file, indent=2)

with open("data1.json", "r") as file:
    data_dump = json.load(file)
    print(data_dump)