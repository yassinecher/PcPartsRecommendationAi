import csv
import os

# Define basic component categories and their corresponding cleaned files
component_categories = {
    "Power Supply": "power_supply_components_cleaned.csv",
    "Storage": "storage_components_cleaned.csv",
    "Video Card": "video_card_components_cleaned.csv",
    "Memory": "memory_components_cleaned.csv",
    "CPU": "cpu_components_cleaned.csv",
    "Motherboard": "motherboard_components_cleaned.csv"
}

# Load existing components from cleaned files into a dictionary
cleaned_components = {}
for category, filename in component_categories.items():
    cleaned_components[category] = set()
    if os.path.exists(filename):
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header if present
            for row in reader:
                if row:  # Ensure row is not empty
                    component_name = row[0].strip().lower()  # Normalize: strip whitespace, lowercase
                    cleaned_components[category].add(component_name)
                    print(f"Loaded {component_name} into {category}")
    else:
        print(f"Warning: {filename} not found in the directory.")

# Read the cleaned_data.csv with utf-8 encoding
builds_data = {}
with open('cleaned_data.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        build_title = row['Build Title']
        component = row['Component']
        product_name = row['Product Name']

        # Only process if the component is a basic category we’re interested in
        if component in component_categories:
            if build_title not in builds_data:
                builds_data[build_title] = {}
            builds_data[build_title][component] = product_name.strip().lower()  # Normalize: strip whitespace, lowercase

# Filter builds that have all basic components present in cleaned files
valid_builds = {}
for build_title, components in builds_data.items():
    is_valid = True
    for category in component_categories.keys():
        if category not in components:
            print(f"Build '{build_title}' is missing {category}")
            is_valid = False
            break
        elif not any(cleaned_name in components[category] for cleaned_name in cleaned_components[category]):
            print(f"Build '{build_title}' has {category} '{components[category]}' not found in {component_categories[category]}")
            is_valid = False
            break
    if is_valid:
        print(f"Build '{build_title}' is valid and will be added to new_builds.csv")
        valid_builds[build_title] = components

# Write valid builds to a new file with utf-8 encoding
with open('new_builds.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['Build Title', 'Component', 'Product Name', 'Link']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for build_title, components in valid_builds.items():
        for component, product_name in components.items():
            # Re-read the original file to get the non-lowercase product name and link
            link = ''
            original_product_name = ''
            with open('cleaned_data.csv', 'r', newline='', encoding='utf-8') as original_file:
                reader = csv.DictReader(original_file)
                for row in reader:
                    if row['Build Title'] == build_title and row['Product Name'].strip().lower() == product_name:
                        link = row['Link']
                        original_product_name = row['Product Name']
                        break
            writer.writerow({
                'Build Title': build_title,
                'Component': component,
                'Product Name': original_product_name,
                'Link': link
            })

print(f"Processing complete. {len(valid_builds)} builds written to 'new_builds.csv'.")