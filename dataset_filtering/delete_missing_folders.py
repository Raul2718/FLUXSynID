import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Keep only the subfolders listed in a text file from a dataset folder."
    )
    parser.add_argument(
        "--txt_path", required=True, type=str, help="Path to the text file containing folder names to keep."
    )
    parser.add_argument(
        "--dataset_path", required=True, type=str, help="Path to the dataset folder containing subfolders."
    )
    args = parser.parse_args()

    if not os.path.exists(args.txt_path):
        raise ValueError(f"File at {args.txt_path} does not exist.")

    folders_to_keep = set()

    # Read the text file and extract folder names
    try:
        with open(args.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                folder_name = parts[0].strip()  # Extract only the folder name
                folders_to_keep.add(folder_name)
    except Exception as e:
        print(f"Error reading file {args.txt_path}: {e}")
        return

    # Get all folders in the dataset directory
    existing_folders = {d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))}

    # Identify folders to remove (those not in the txt file)
    folders_to_remove = existing_folders - folders_to_keep

    # Remove the folders
    for folder in folders_to_remove:
        folder_path = os.path.join(args.dataset_path, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Removed folder: {folder_path}")
            except Exception as e:
                print(f"Error removing folder {folder_path}: {e}")
        else:
            print(f"Folder not found or already removed: {folder_path}")


if __name__ == "__main__":
    main()
