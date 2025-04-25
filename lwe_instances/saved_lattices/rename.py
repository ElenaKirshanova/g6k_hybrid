import os
import re
import sys

def rename_files(directory):
    # Regular expression to match filenames of the form:
    # lwe_instance_{n}_{q}_{p}_{k}_{i}.pkl
    pattern = re.compile(r'^lwe_instance_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)')
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            n, q, p, k, i = match.groups()
            new_filename = f"lwe_instance_binomial_{n}_{q}_{p}_{i}.pkl"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            print(f"Renaming '{filename}' to '{new_filename}'")
            os.rename(old_path, new_path)
        else:
            print(f"Skipping file: {filename}")

if __name__ == '__main__':
    # If a directory is provided as a command line argument, use it; otherwise, use the current directory.
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    rename_files(folder)
