import shutil
from pathlib import Path


def safe_move_folder(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        print(f"Source not found: {src}")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        print(f"Destination already exists, skipping: {dst}")
        return

    print(f"Moving {src} → {dst}")
    shutil.move(str(src_path), str(dst_path))


def safe_create_folder(path):
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")


# Base directory (assuming run from project root)
BASE_DIR = Path(__file__).parent.resolve()

# Create output folder for ComfyUI
safe_create_folder(BASE_DIR / "ComfyUI" / "output" / "exp_data")

# Move Arc2Face model
safe_move_folder(BASE_DIR / "models" / "Arc2Face" / "antelopev2", BASE_DIR / "Arc2Face" / "models" / "antelopev2")

# Move ComfyUI insightface model
safe_move_folder(
    BASE_DIR / "models" / "ComfyUI" / "insightface" / "models",
    BASE_DIR / "ComfyUI" / "models" / "insightface" / "models",
)
