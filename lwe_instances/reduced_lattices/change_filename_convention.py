# usage example
#python change_filename_convention.py "report_prehyb_138_3329_binomial_2.0000_" "_6_57_58.pkl" 57

#!/usr/bin/env python3
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Rename files of the form <prefix><seed><suffix> by inserting a beta value before the .pkl extension."
    )
    parser.add_argument("prefix", help="Prefix of the files (e.g., 'g6kdump_100_20_norm_0.1234_').")
    parser.add_argument("suffix", help="Suffix of the files (e.g., '_0.5_42.pkl').")
    parser.add_argument("beta", help="Beta value to insert.")
    args = parser.parse_args()

    prefix = args.prefix
    suffix = args.suffix
    beta = args.beta

    # Pattern to match files of the form <prefix>*<suffix>
    pattern = f"{prefix}*{suffix}"
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    for file in files:
        # Extract the seed by stripping the prefix and suffix from the filename.
        base = os.path.basename(file)
        if not ( (base.startswith(prefix) or base.startswith("./"+prefix)) and base.endswith(suffix)):
            # print((base.startswith(prefix) or base.startswith("./"+prefix)), base.endswith(suffix))
            print(prefix)
            continue

        seed = base[len(prefix):-len(suffix)]

        # Build new filename: insert _{beta} before the .pkl extension.
        if base.endswith(".pkl"):
            new_base = base[:-4] + f"_{beta}.pkl"
        else:
            new_base = base + f"_{beta}"

        new_file = os.path.join(os.path.dirname(file), new_base)
        print(f"Renaming '{file}' to '{new_file}'")
        os.rename(file, new_file)

if __name__ == "__main__":
    main()
