import os

os.environ["HF_HOME"] = "./models/huggingface"

import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from torchvision.transforms import ToPILImage
import argparse
from PIL import Image, ImageOps, ImageSequence
import numpy as np
from tqdm import tqdm
from loc_utils.capture_outputs import capture_all_output
import json
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate live capture images.", add_help=False
    )  # Disable default help to avoid conflicts with ComfyUI's help

    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument(
        "--num_live_imgs", type=int, default=1, help="Number of different live images to generate per identity."
    )
    parser.add_argument(
        "--no_pulid", action="store_true", help="Disable use of PuLID (only expression change will be done)"
    )
    parser.add_argument(
        "--no_same_seed_and_prompt",
        action="store_false",
        help="Disable using the same seed and modified prompt with PuLID.",
    )

    # Use `parse_known_args()` to separate unknown arguments (ComfyUI's args)
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def clean_caption(caption):
    # Match and capture the subject description while removing the unwanted prefix
    pattern = r"^s7ll2f4h, Photograph of (.*?). The subject has a neutral face expression\."

    # Replace with the new background scene while keeping the subject descriptor
    caption = re.sub(
        pattern,
        (
            r"A colored photograph of a \1 inside a modern airport terminal, "
            r"with large glass windows, steel beams, and rows of seats in the background. "
            r"The space is well-lit, with visible signs and a long corridor."
        ),
        caption,
        count=1,
    )

    # Remove the ending part
    caption = re.sub(
        (
            r"Photo captures the subject from the neck up with grey clothing slightly visible around the neck.  "
            r"White background.$"
        ),
        "",
        caption,
    ).strip()

    return caption


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.insert(0, comfyui_path)  # Insert at the beginning to prioritize it
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from ComfyUI.main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from ComfyUI.utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import ComfyUI.execution as execution
    from ComfyUI.nodes import init_extra_nodes
    import ComfyUI.server as server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


# Modified ComfyUI class for specifying directory from which images are loaded
class LoadImageEdited:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image, dir):
        image_path = folder_paths.get_annotated_filepath(image, default_dir=dir)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)


def main(dataset_dir, num_live_imgs, no_pulid, no_same_seed_and_prompt):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    import_custom_nodes()
    with torch.inference_mode():
        loadimage = LoadImageEdited()
        expressioneditor = NODE_CLASS_MAPPINGS["ExpressionEditor"]()

        if not no_pulid:
            vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            vaeloader_3 = vaeloader.load_vae(vae_name="ae.safetensors")

            pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
            pulidfluxmodelloader_4 = pulidfluxmodelloader.load_model(pulid_file="pulid_flux_v0.9.1.safetensors")

            pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()

            local_eva_clip_path = "./models/huggingface/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt"
            pulidfluxevacliploader_5 = pulidfluxevacliploader.load_eva_clip(local_eva_clip_path)

            pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
            pulidfluxinsightfaceloader_6 = pulidfluxinsightfaceloader.load_insightface(provider="CPU")

            unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            unetloader_25 = unetloader.load_unet(unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn_fast")

            dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            dualcliploader_26 = dualcliploader.load_clip(
                clip_name1="clip_l.safetensors",
                clip_name2="t5xxl_fp16.safetensors",
                type="flux",
            )

            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            loraloader_61 = loraloader.load_lora(
                lora_name="flux1-canny-dev-lora.safetensors",
                strength_model=1,
                strength_clip=1,
                model=get_value_at_index(unetloader_25, 0),
                clip=get_value_at_index(dualcliploader_26, 0),
            )

            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

            fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()

            aio_preprocessor = NODE_CLASS_MAPPINGS["AIO_Preprocessor"]()

            instructpixtopixconditioning = NODE_CLASS_MAPPINGS["InstructPixToPixConditioning"]()

            applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        failed_log_path = "dataset_filtering/failed_live.txt"
        # Load existing failed entries
        if os.path.exists(failed_log_path):
            with open(failed_log_path, "r") as f:
                failed_entries = set(f.read().splitlines())
        else:
            failed_entries = set()

        first_pass_flag = True  # Added due to a bug when using CUDA 12.8
        print(
            (
                "WARNING: All outputs are now being captured, including errors. "
                "If issues arise, set capture_all_output(disable=True) for debugging."
            )
        )

        for folder in tqdm(os.listdir(dataset_dir)):
            # NOTE: This captures all output, including errors. If issues arise, set disable=True for debugging
            with capture_all_output(disable=False):
                folder_path = os.path.join(dataset_dir, folder)

                # Load document image caption and seed
                if not no_pulid and not no_same_seed_and_prompt:
                    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
                    if len(json_files) != 1:
                        raise ValueError(
                            f"Expected exactly one .json file in '{folder_path}', but found {len(json_files)}."
                        )

                    json_file = json_files[0]
                    json_file_path = os.path.join(folder_path, json_file)

                    with open(json_file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        caption = data["prompt"].strip()
                        caption = clean_caption(caption)
                        seed = data["seed"]

                doc_img_files = [f for f in os.listdir(folder_path) if f.endswith("_doc.png")]
                if len(doc_img_files) != 1:
                    raise ValueError(
                        f"Expected exactly one _doc.png file in '{folder_path}', but found {len(doc_img_files)}."
                    )

                doc_img_file = doc_img_files[0]

                for live_img_num in range(num_live_imgs):
                    # Path for saving LivePortrait-based imaged
                    output_image_e_path = os.path.join(
                        folder_path, doc_img_file.removesuffix("_doc.png") + f"_live_{live_img_num}_e_d1.png"
                    )

                    # Path for saving PuLID-based imaged
                    if not no_pulid:
                        output_image_p_path = os.path.join(
                            folder_path, doc_img_file.removesuffix("_doc.png") + f"_live_{live_img_num}_p_d1.png"
                        )

                    # Skip if images already exist
                    if no_pulid and os.path.exists(output_image_e_path):
                        continue
                    elif not no_pulid and os.path.exists(
                        output_image_p_path
                    ):  # If PuLID image exists, then LivePortrait image also already exists
                        continue

                    loadimage_7 = loadimage.load_image(image=doc_img_file, dir=folder_path)

                    # Apply LivePortrait by randomly choosing constrained values for pose and expression editing
                    expressioneditor_71 = expressioneditor.run(
                        rotate_pitch=random.uniform(-15, 15),
                        rotate_yaw=random.uniform(-10, 10),
                        rotate_roll=random.uniform(-15, 15),
                        blink=random.uniform(-3, 5),
                        eyebrow=random.uniform(-10, 15),
                        wink=0,
                        pupil_x=random.uniform(-10, 10),
                        pupil_y=random.uniform(-15, 15),
                        aaa=0,
                        eee=0,
                        woo=0,
                        smile=random.uniform(-0.3, 0.7),
                        src_ratio=1,
                        sample_ratio=1,
                        sample_parts="OnlyExpression",
                        crop_factor=2,
                        src_image=get_value_at_index(loadimage_7, 0),
                    )

                    ToPILImage()(get_value_at_index(expressioneditor_71, 0).squeeze(0).permute(2, 0, 1)).save(
                        output_image_e_path
                    )

                    # Continue if only LivePortrait images are being generated
                    if no_pulid:
                        continue

                    # first_pass_flag ensures that same prompt is not encoded everytime a new image is generated. This
                    # had to be done here due to a bug when using CUDA 12.8
                    if first_pass_flag:
                        first_pass_flag = False

                        # Use a generic prompt if the document image prompt is not reused
                        if no_same_seed_and_prompt:
                            cliptextencode_8 = cliptextencode.encode(
                                text=(
                                    "Photo of a person inside inside a modern airport terminal, "
                                    "with large glass windows, steel beams, and rows of seats in the background. "
                                    "The space is well-lit, with visible signs and a long corridor."
                                ),
                                clip=get_value_at_index(loraloader_61, 1),
                            )
                            fluxguidance_46 = fluxguidance.append(
                                guidance=4, conditioning=get_value_at_index(cliptextencode_8, 0)
                            )

                        # Encode negative (unused due to FLUX not having true CFG) prompt only once
                        cliptextencode_47 = cliptextencode.encode(text="", clip=get_value_at_index(loraloader_61, 1))

                    # Encode positive prompt
                    if not no_same_seed_and_prompt and live_img_num == 0:
                        cliptextencode_8 = cliptextencode.encode(
                            text=caption, clip=get_value_at_index(loraloader_61, 1)
                        )

                        fluxguidance_46 = fluxguidance.append(
                            guidance=4, conditioning=get_value_at_index(cliptextencode_8, 0)
                        )

                    # Extract line art edges from the LivePortrait-based image
                    aio_preprocessor_63 = aio_preprocessor.execute(
                        preprocessor="LineArtPreprocessor",
                        resolution=get_value_at_index(expressioneditor_71, 0).squeeze(0).shape[1],
                        image=get_value_at_index(expressioneditor_71, 0),
                    )

                    # Condition FLUX using extracted edges. This encodes the edges and concatenates the input latent img
                    instructpixtopixconditioning_60 = instructpixtopixconditioning.encode(
                        positive=get_value_at_index(fluxguidance_46, 0),
                        negative=get_value_at_index(cliptextencode_47, 0),
                        vae=get_value_at_index(vaeloader_3, 0),
                        pixels=get_value_at_index(aio_preprocessor_63, 0),
                    )

                    # Sometimes PuLID can fail to detect faces and this is logged
                    try:
                        # Apply PuLID using default parameters
                        applypulidflux_9 = applypulidflux.apply_pulid_flux(
                            weight=1,
                            start_at=0,
                            end_at=1,
                            fusion="mean",
                            fusion_weight_max=1,
                            fusion_weight_min=0,
                            train_step=1000,
                            use_gray=True,
                            model=get_value_at_index(loraloader_61, 0),
                            pulid_flux=get_value_at_index(pulidfluxmodelloader_4, 0),
                            eva_clip=get_value_at_index(pulidfluxevacliploader_5, 0),
                            face_analysis=get_value_at_index(pulidfluxinsightfaceloader_6, 0),
                            image=get_value_at_index(loadimage_7, 0),
                            prior_image=get_value_at_index(expressioneditor_71, 0),
                            unique_id=2507743956387314891,
                        )
                    except ValueError:
                        log_entry = f"{os.path.basename(folder_path)}, PuLID"
                        if log_entry not in failed_entries:  # Avoid duplicate logging
                            with open(failed_log_path, "a") as f:
                                f.write(log_entry + "\n")
                            failed_entries.add(log_entry)  # Update the set
                        break

                    # Run diffusion model with ksampler to generate PuLID-based image
                    ksampler_45 = ksampler.sample(
                        seed=seed if not no_same_seed_and_prompt else random.randint(1, 2**64),
                        steps=20,
                        cfg=1,
                        sampler_name="dpmpp_2m",
                        scheduler="beta",
                        denoise=1,
                        model=get_value_at_index(applypulidflux_9, 0),
                        positive=get_value_at_index(instructpixtopixconditioning_60, 0),
                        negative=get_value_at_index(instructpixtopixconditioning_60, 1),
                        latent_image=get_value_at_index(instructpixtopixconditioning_60, 2),
                    )

                    # Decode latent image
                    vaedecode_17 = vaedecode.decode(
                        samples=get_value_at_index(ksampler_45, 0),
                        vae=get_value_at_index(vaeloader_3, 0),
                    )

                    ToPILImage()(get_value_at_index(vaedecode_17, 0).squeeze(0).permute(2, 0, 1)).save(
                        output_image_p_path
                    )


if __name__ == "__main__":
    args, remaining_args = parse_args()

    # Merge back the remaining arguments into `sys.argv`
    sys.argv = [sys.argv[0]] + remaining_args  # Restore unknown args for ComfyUI

    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    from ComfyUI.nodes import NODE_CLASS_MAPPINGS
    import ComfyUI.folder_paths as folder_paths
    import ComfyUI.node_helpers as node_helpers

    main(
        dataset_dir=args.dataset_dir,
        num_live_imgs=args.num_live_imgs,
        no_pulid=args.no_pulid,
        no_same_seed_and_prompt=args.no_same_seed_and_prompt,
    )
