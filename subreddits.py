import csv
import json
from collections import OrderedDict

# Initialize empty lists to store unique categories
columns = {
    'second_column': [],
    'third_column': [],
    'fourth_column': [],
    'fifth_column': [],
    'sixth_column': [],
    'seventh_column': [],
    'eighth_column': []
}

# Read the CSV file
with open('subreddits.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    # Process each row
    for row in csv_reader:
        # Skip empty rows
        if len(row) < 8:  # Ensure row has all columns (including subreddit name)
            continue
            
        # Add non-empty values to respective lists (columns 2-8)
        for i in range(1, 8):
            if row[i].strip():  # Check if value is non-empty
                columns[f'{"second" if i==1 else "third" if i==2 else "fourth" if i==3 else "fifth" if i==4 else "sixth" if i==5 else "seventh" if i==6 else "eighth"}_column'].append(row[i])

# Remove duplicates while preserving order
for key in columns:
    columns[key] = list(OrderedDict.fromkeys(columns[key]))

# Save results to JSON file
with open('subreddits.json', 'w', encoding='utf-8') as json_file:
    json.dump(columns, json_file, indent=4)

# Optional: Print the results
for key, value in columns.items():
    print(f"{key} =", value) 