import os
import re
import csv
from collections import defaultdict


def remove_output_file():
    try:
        os.remove('output.csv')
        print("output.csv removed successfully.")
    except FileNotFoundError:
        print("output.csv does not exist.")

def remove_zero_columns(filename):
    temp_filename = f'temp_{filename}'
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        rows = list(reader)
        for field in fieldnames:
            if all(row[field] == '0' or row[field] == '' for row in rows[1:]):
                for row in rows:
                    del row[field]
        with open(temp_filename, 'w', newline='') as temp_csvfile:
            writer = csv.DictWriter(temp_csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temp_filename, filename)


# Function to write data to a CSV file
def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['System', 'Mode', 'X-axis', 'Comp', 'Decomp', 'Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Function to extract relevant information from each section
def extract_info(section):
    info = {}
    lines = section.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip()] = value.strip()
    return info

# Function to extract mode, system, and x-axis from section name
def extract_mode_system_xaxis(section_name):
    # Define regex pattern to extract mode, system, and x-axis
    pattern = r'Geometric mean of median throughputs for (\w+)-(\w+)-(\w+)\.exe:'
    match = re.match(pattern, section_name)
    if match:
        mode = match.group(1)
        system = match.group(2)
        x_axis = match.group(3)
        return mode, system, x_axis
    else:
        pattern2 = r'Geometric mean of median ratios for (\w+)-(\w+)-(\w+)\.exe:'
        match2 = re.match(pattern2, section_name)
        if match2:
            mode = match2.group(1)
            system = match2.group(2)
            x_axis = match2.group(3)
            return mode, system, x_axis

import os

def combine_text_files(directory):
    # Specify the directory containing the text files
    input_directory = os.path.join(os.getcwd(), directory)
    
    # Check if the directory exists
    if not os.path.exists(input_directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Initialize an empty string to store the combined content
    combined_content = ""
    
    # Iterate over all files in the directory
    for filename in os.listdir(input_directory):
        # Check if the file is a text file
        if filename.endswith(".txt"):
            filepath = os.path.join(input_directory, filename)
            # Open and read the content of the text file
            with open(filepath, "r") as file:
                file_content = file.read()
                # Append the content to the combined content string
                combined_content += file_content + "\n"  # Add a newline between each file's content
    
    # Write the combined content to a new file
    output_file = os.path.join("combined.txt")
    with open(output_file, "w") as combined_file:
        combined_file.write(combined_content)
    
    print(f"Combined content has been saved to '{output_file}'.")
    
def delete_text_files(directory):
    # Specify the directory containing the text files
    input_directory = os.path.join(os.getcwd(), directory)
    
    # Check if the directory exists
    if not os.path.exists(input_directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Iterate over all files in the directory
    for filename in os.listdir(input_directory):
        # Check if the file is a text file
        if filename.endswith(".txt"):
            filepath = os.path.join(input_directory, filename)
            os.remove(filepath)
       
   

# Call the function with the directory name
combine_text_files("single_inputs")
delete_text_files("single_inputs")
# Read the contents of the combined.txt file
with open('combined.txt', 'r') as file:
    data = file.read()

# Split the data into sections based on the double newline character
sections = data.strip().split('\n\n')
decomp = None
comp = None
ratio = None
# Convert defaultdict to list of dictionaries
data_list = []

# Extract information from each section
for section in sections:
    lines = section.strip().split('\n')
    if lines:
        section_name = lines[0]
        section_data = extract_info('\n'.join(lines[1:]))
        mode, system, x_axis = extract_mode_system_xaxis(section_name)
       
        # Extracting x-axis or y-axis based on the last line of the section
        last_line = lines[-1].strip()
        if x_axis == 'compress':
            if 'throughput' in last_line:
                comp = float(last_line.split(':')[-1].strip())
                ratio = 0
                decomp = 0
            elif 'ratio' in last_line:
                ratio = float(last_line.split(':')[-1].strip())
                decomp = 0
                comp = 0

        else:
            decomp = float(last_line.split(':')[-1].strip())
            comp = 0
            ratio = 0

        data_list.append({
            'System': system,
            'Mode': mode,
            'X-axis': x_axis,
            'Comp': comp,
            'Decomp': decomp,
            'Ratio': ratio
        })

# Write the data to a CSV file
write_to_csv('output.csv', data_list)



input_file = 'output.csv'

cpu_data = defaultdict(list)
gpu_data = defaultdict(list)

# Read the input CSV file
with open(input_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['System'] == 'cpu':
            cpu_data[row['X-axis']].append(row)
        elif row['System'] == 'gpu':
            gpu_data[row['X-axis']].append(row)

# Write CPU data to CSV files with the same X-axis
for x_axis, data in cpu_data.items():
    cpu_output_file = f'cpu_output_{x_axis}.csv'
    mode_data = defaultdict(dict)
    for row in data:
        mode = row['Mode']
        if mode not in mode_data:
            mode_data[mode] = row
        else:
            for key, value in row.items():
                if value != '0':
                    mode_data[mode][key] = value
    
    with open(cpu_output_file, 'w', newline='') as csvfile:
        fieldnames = ['System', 'Mode', 'X-axis', 'Comp', 'Decomp', 'Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for mode, row in mode_data.items():
            writer.writerow(row)

# Write GPU data to CSV files with the same X-axis
for x_axis, data in gpu_data.items():
    gpu_output_file = f'gpu_output_{x_axis}.csv'
    mode_data = defaultdict(dict)
    for row in data:
        mode = row['Mode']
        if mode not in mode_data:
            mode_data[mode] = row
        else:
            for key, value in row.items():
                if value != '0':
                    mode_data[mode][key] = value
    
    with open(gpu_output_file, 'w', newline='') as csvfile:
        fieldnames = ['System', 'Mode', 'X-axis', 'Comp', 'Decomp', 'Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for mode, row in mode_data.items():
            writer.writerow(row)

print("Separation and combination completed successfully.")
def copy_ratios(compress_file, decompress_file):
    # Read the 'Ratio' values from the compress file
    compress_ratios = {}
    with open(compress_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            compress_ratios[row['Mode']] = row['Ratio']

    # Copy the 'Ratio' values to the decompress file
    updated_rows = []
    with open(decompress_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mode = row['Mode']
            if mode in compress_ratios:
                row['Ratio'] = compress_ratios[mode]
            updated_rows.append(row)

    # Write the updated rows to the decompress file
    with open(decompress_file, 'w', newline='') as csvfile:
        fieldnames = ['System', 'Mode', 'X-axis', 'Comp', 'Decomp', 'Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Ratio values copied successfully from {compress_file} to {decompress_file}.")

copy_ratios('gpu_output_compress.csv', 'gpu_output_decompress.csv')
copy_ratios('cpu_output_compress.csv', 'cpu_output_decompress.csv')


# Remove output.csv
remove_output_file()

# Remove zero-value columns from CPU files
remove_zero_columns('cpu_output_compress.csv')
remove_zero_columns('cpu_output_decompress.csv')

# Remove zero-value columns from GPU files
remove_zero_columns('gpu_output_compress.csv')
remove_zero_columns('gpu_output_decompress.csv')
