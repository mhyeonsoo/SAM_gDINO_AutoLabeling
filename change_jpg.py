#!/usr/bin/env python3
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Recursively rename files in target_dir. If a file's name does not end with '.jpg' (case-insensitive), rename it to the folder's name followed by a sequential number and '.jpg'. Files that already end with '.jpg' are left unchanged."
    )
    parser.add_argument("target_dir", help="Top-level directory containing files to rename.")
    args = parser.parse_args()

    target_dir = args.target_dir

    # Recursively traverse all folders in the target directory
    for root, dirs, files in os.walk(target_dir):
        if not files:
            continue

        # Get the current folder name (use absolute path if basename is empty)
        folder_name = os.path.basename(root)
        if not folder_name:
            folder_name = os.path.basename(os.path.abspath(root))

        # Sort files and rename them sequentially
        counter = 1
        for file in sorted(files):
            # Check if the file already ends with ".jpg" (case-insensitive)
            if file.lower().endswith(".jpg"):
                continue  # leave it as is

            # Create new filename using the folder name as a prefix
            new_name = f"{folder_name}{counter}.jpg"
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, new_name)

            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Error renaming {old_path} to {new_path}: {e}")
            counter += 1

    print("All files have been successfully renamed where needed.")

if __name__ == "__main__":
    main()
