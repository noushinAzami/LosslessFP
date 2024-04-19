#!/usr/bin/env python3
import sys
import subprocess

# List of eight input executable files
exe_files = [
    "speed-gpu-compress.exe",
    "speed-gpu-decompress.exe",
    "ratio-gpu-compress.exe",
    "ratio-gpu-decompress.exe",
    "speed-cpu-compress.exe",
    "speed-cpu-decompress.exe",
    "ratio-cpu-compress.exe",
    "ratio-cpu-decompress.exe"
]


# Function to run comp-run and decomp-run scripts on an exe file
def run_scripts(exe_file):
    # Run comp-run script
    subprocess.run(["./comp-run.py","single_inputs", "single_src/bin", exe_file])
    
    # Run decomp-run script
    subprocess.run(["./decomp-run.py", "single_inputs", "single_src/bin", exe_file])

# Iterate over each exe file and run the scripts
for exe_file in exe_files:
    run_scripts(exe_file)

