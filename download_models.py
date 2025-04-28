from huggingface_hub import hf_hub_download, snapshot_download, login
from insightface.utils.storage import download
import os
import shutil


HF_TOKEN = "PUT_YOUR_TOKEN_HERE"

MODELS_FULL = [
    ("models/ComfyUI/vae/", "black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
    ("models/ComfyUI/loras/", "black-forest-labs/FLUX.1-Canny-dev-lora", "flux1-canny-dev-lora.safetensors"),
    ("models/ComfyUI/clip/", "comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
    ("models/ComfyUI/clip/", "comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    ("models/ComfyUI/liveportrait/", "Kijai/LivePortrait_safetensors", "appearance_feature_extractor.safetensors"),
    ("models/ComfyUI/liveportrait/", "Kijai/LivePortrait_safetensors", "motion_extractor.safetensors"),
    ("models/ComfyUI/liveportrait/", "Kijai/LivePortrait_safetensors", "spade_generator.safetensors"),
    ("models/ComfyUI/liveportrait/", "Kijai/LivePortrait_safetensors", "stitching_retargeting_module.safetensors"),
    ("models/ComfyUI/liveportrait/", "Kijai/LivePortrait_safetensors", "warping_module.safetensors"),
    ("models/ComfyUI/loras/", "Raul2718/CFD-FLUX.1-dev-LoRA", "cfd-lora.safetensors"),
    ("models/ComfyUI/pulid/", "guozinan/PuLID", "pulid_flux_v0.9.1.safetensors"),
    ("models/ComfyUI/ultralytics/", "Bingsu/adetailer", "face_yolov8n.pt"),
    ("models/ComfyUI/unet/", "Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
    ("models/Arc2Face/", "FoivosPar/Arc2Face", "arc2face/config.json"),
    ("models/Arc2Face/", "FoivosPar/Arc2Face", "arc2face/diffusion_pytorch_model.safetensors"),
    ("models/Arc2Face/", "FoivosPar/Arc2Face", "encoder/config.json"),
    ("models/Arc2Face/", "FoivosPar/Arc2Face", "encoder/pytorch_model.bin"),
    ("models/Arc2Face/antelopev2", "FoivosPar/Arc2Face", "arcface.onnx"),
    ("models/huggingface/sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", "model_index.json"),
    ("models/huggingface/EVA-CLIP/", "QuanSun/EVA-CLIP", "EVA02_CLIP_L_336_psz14_s6B.pt"),
    ("models/huggingface/sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", "tokenizer/merges.txt"),
    ("models/huggingface/sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", "tokenizer/vocab.json"),
    ("models/huggingface/sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", "vae/config.json"),
    (
        "models/huggingface/sd15",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "feature_extractor/preprocessor_config.json",
    ),
    (
        "models/huggingface/sd15",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "scheduler/scheduler_config.json",
    ),
    (
        "models/huggingface/sd15",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "tokenizer/tokenizer_config.json",
    ),
    (
        "models/huggingface/sd15",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "vae/diffusion_pytorch_model.safetensors",
    ),
]

MODELS_SPLIT = [
    ("models/huggingface/Qwen2.5/", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"),
]

# Authenticate
login(token=HF_TOKEN)


def download_file(target_dir, repo_id, filename):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading {filename} from {repo_id} to {target_dir}")
    try:
        filepath = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=target_dir, token=HF_TOKEN, local_dir_use_symlinks=False
        )
        print(f"Downloaded: {filepath}")
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}")


def download_full_model(target_dir, repo_id):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading full snapshot of {repo_id} to {target_dir}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=target_dir, token=HF_TOKEN, local_dir_use_symlinks=False)
        print(f"Downloaded full model to: {target_dir}")
    except Exception as e:
        print(f"Failed to download full model from {repo_id}: {e}")


def download_antelopev2(target_dir, remove_glintr=False):
    expected_file = os.path.join(target_dir, "1k3d68.onnx")

    if os.path.exists(expected_file):
        print(f"antelopev2 already exists, skipping download.")
        return

    print("Downloading antelopev2...")

    nested_base = download(sub_dir="", name="antelopev2", force=True, root=os.path.dirname(target_dir))
    nested_dir = os.path.join(nested_base, "antelopev2")

    if os.path.isdir(nested_dir):
        os.makedirs(target_dir, exist_ok=True)
        for file_name in os.listdir(nested_dir):
            src = os.path.join(nested_dir, file_name)
            dst = os.path.join(target_dir, file_name)
            shutil.move(src, dst)
        os.rmdir(nested_dir)

    zip_file = os.path.join(os.path.dirname(target_dir), "antelopev2.zip")
    if os.path.exists(zip_file):
        os.remove(zip_file)

    if remove_glintr:
        glintr = os.path.join(target_dir, "glintr100.onnx")
        if os.path.exists(glintr):
            os.remove(glintr)
            print("Removed glintr100.onnx")

    print(f"antelopev2 is ready at: {target_dir}")


def download_buffalo_l(target_dir):
    expected_file = os.path.join(target_dir, "1k3d68.onnx")

    if os.path.exists(expected_file):
        print(f"buffalo_l already exists, skipping download.")
        return

    print("Downloading buffalo_l...")

    nested_base = download(sub_dir="", name="buffalo_l", force=True, root=os.path.dirname(target_dir))
    nested_dir = os.path.join(nested_base, "buffalo_l")

    if os.path.isdir(nested_dir):
        os.makedirs(target_dir, exist_ok=True)
        for file_name in os.listdir(nested_dir):
            src = os.path.join(nested_dir, file_name)
            dst = os.path.join(target_dir, file_name)
            shutil.move(src, dst)
        os.rmdir(nested_dir)

    zip_file = os.path.join(os.path.dirname(target_dir), "buffalo_l.zip")
    if os.path.exists(zip_file):
        os.remove(zip_file)

    print(f"buffalo_l is ready at: {target_dir}")


def main():
    # Download individual files
    for target_dir, repo_id, filename in MODELS_FULL:
        download_file(target_dir, repo_id, filename)

    # Download full snapshot models
    for target_dir, repo_id in MODELS_SPLIT:
        download_full_model(target_dir, repo_id)

    # antelopev2 for ComfyUI
    download_antelopev2("models/ComfyUI/insightface/models/antelopev2", remove_glintr=False)

    # buffalo_l for ComfyUI
    download_buffalo_l("models/ComfyUI/insightface/models/buffalo_l")

    # antelopev2 for Arc2Face
    download_antelopev2("models/Arc2Face/antelopev2", remove_glintr=True)


if __name__ == "__main__":
    main()
