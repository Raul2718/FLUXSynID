import os
import torch
import numpy as np
from torchvision.transforms import ToPILImage
import argparse
from tqdm import tqdm

# from utils.capture_outputs import capture_all_output
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompts with specified parameters.", add_help=False)
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument(
        "--num_live_imgs", type=int, default=1, help="Number of different live images to generate per identity."
    )
    return parser.parse_args()


def main(dataset_dir, num_live_imgs):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    print("Initializing models...")

    # Initialize Arc2Face and Stable Diffusion
    base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    encoder = CLIPTextModelWrapper.from_pretrained("models", subfolder="encoder", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained("models", subfolder="arc2face", torch_dtype=torch.float16)
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model, text_encoder=encoder, unet=unet, torch_dtype=torch.float16, safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")

    # Initialize Face Analysis model
    app = FaceAnalysis(name="antelopev2", root="./", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("Models initialized successfully.")
    print(
        "WARNING: All outputs are now being captured, including errors. If issues arise, set capture_all_output(disable=True) for debugging."
    )

    for folder in tqdm(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)

        if not os.path.isdir(folder_path):
            continue  # Skip non-folder files

        # with capture_all_output(disable=False):
        doc_img_files = [f for f in os.listdir(folder_path) if f.endswith("_doc.png")]
        if len(doc_img_files) != 1:
            raise ValueError(f"Expected exactly one _doc.png file in '{folder_path}', but found {len(doc_img_files)}.")

        doc_img_file = doc_img_files[0]
        doc_img_path = os.path.join(folder_path, doc_img_file)
        img = np.array(Image.open(doc_img_path))[:, :, ::-1]

        # Extract face embedding
        faces = app.get(img)
        if len(faces) == 0:
            raise ValueError(f"No face detected in '{doc_img_path}'.")

        faces = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # Select largest face
        id_emb = torch.tensor(faces["embedding"], dtype=torch.float16)[None].cuda()
        id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)  # Normalize embedding
        id_emb = project_face_embs(pipeline, id_emb)  # Pass through encoder

        for live_img_num in range(num_live_imgs):
            output_image_path = os.path.join(
                folder_path, doc_img_file.removesuffix("_doc.png") + f"_live_{live_img_num}_a_d1.png"
            )

            if os.path.exists(output_image_path):
                continue  # Skip if already exists

            # Generate image
            images = pipeline(
                prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=1
            ).images

            # Save generated image
            images[0].save(output_image_path)

    print("Image generation completed.")


if __name__ == "__main__":
    args = parse_args()
    main(dataset_dir=args.dataset_dir, num_live_imgs=args.num_live_imgs)
