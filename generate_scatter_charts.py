#!/usr/bin/python3 -u

import subprocess
import os

# Run scripts/pareto.py
subprocess.run(['python3', 'scripts/pareto.py'])

# Run scripts/showcpu.py
subprocess.run(['python3', 'scripts/showcpu.py'])

# Run scripts/showgpu.py
subprocess.run(['python3', 'scripts/showgpu.py'])

# Get the current directory
current_directory = os.getcwd()

# List all files in the current directory
files = os.listdir(current_directory)

# Iterate over each file
for file in files:
    # Check if the file ends with ".csv"
    if file.endswith(".csv") or file.endswith(".txt"):
        # Construct the absolute file path
        file_path = os.path.join(current_directory, file)
        # Remove the file
        os.remove(file_path)
