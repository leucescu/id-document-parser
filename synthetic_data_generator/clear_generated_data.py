from pathlib import Path

DATA_DIRS = ["data/train", "data/validation"]
FILE_EXTENSIONS = [".png", ".json"]

def clear_files(directories, extensions):
    for dir_path in directories:
        path_obj = Path(dir_path)
        if not path_obj.exists():
            print(f"Directory not found: {dir_path}")
            continue

        deleted = 0
        for file in path_obj.iterdir():
            if file.suffix in extensions:
                file.unlink()
                deleted += 1
        print(f"Cleared {deleted} files from: {dir_path}")

if __name__ == "__main__":
    clear_files(DATA_DIRS, FILE_EXTENSIONS)
