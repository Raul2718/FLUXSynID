import os

os.environ["HF_HOME"] = "./models/huggingface"

import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy
import inflect
from tqdm import tqdm
import argparse
import uuid
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompts with specified attributes.")

    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset will be saved.")
    parser.add_argument("--number_of_prompts", type=int, required=True, help="Number of prompts to generate.")
    parser.add_argument(
        "--file_prob_txt",
        type=str,
        default="attributes/attributes/file_probabilities.json",
        help="Path to file probabilities JSON.",
    )
    parser.add_argument(
        "--attribute_clash_txt",
        type=str,
        default="attributes/attributes/attribute_clashes.json",
        help="Path to attribute clashes JSON.",
    )

    return parser.parse_args()


def contains_non_english(text):
    """Detects if the generated text contains non-English characters."""
    return bool(re.search(r"[^\x00-\x7F]", text))  # Checks for any non-ASCII character


def load_attributes_and_clashes(file_prob_txt, attribute_clash_txt):
    def validate_clash_rules(attributes, clash_rules):
        for rule in clash_rules:
            fixed_file = rule["fixed_file"]
            secondary_file = rule["secondary_file"]
            fixed_attributes = rule["fixed_attributes"]
            secondary_attributes = rule["secondary_attributes"]

            # Check if files exist
            if fixed_file not in attributes:
                raise ValueError(f"Fixed file '{fixed_file}' does not exist in attributes.")
            if secondary_file not in attributes:
                raise ValueError(f"Secondary file '{secondary_file}' does not exist in attributes.")

            # Check if fixed attributes exist
            if fixed_file in attributes:
                for attr in fixed_attributes:
                    if attr not in attributes[fixed_file]["attribute_probabilities"]:
                        raise ValueError(f"Attribute '{attr}' in fixed file '{fixed_file}' does not exist.")

            # Check if secondary attributes exist
            if secondary_file in attributes:
                for attr in secondary_attributes:
                    if attr not in attributes[secondary_file]["attribute_probabilities"]:
                        raise ValueError(f"Attribute '{attr}' in secondary file '{secondary_file}' does not exist.")

    # Load probabilities from file
    with open(file_prob_txt, "r") as f:
        file_probabilities = json.load(f)

    # Load attribute clashes
    with open(attribute_clash_txt, "r") as f:
        clash_rules = json.load(f)

    attributes = {}
    for filename, prob in file_probabilities.items():
        if prob > 0:  # Only consider files with non-zero probability
            filepath = os.path.join(os.path.dirname(file_prob_txt), filename)
            with open(filepath, "r") as f:
                lines = f.readlines()
                attribute_probabilities = {}
                for line in lines:
                    key, value = line.strip().replace("\n", "").split(",")
                    attribute_probabilities[key] = float(value)
                attributes[filename] = {"prob_of_this_file": prob, "attribute_probabilities": attribute_probabilities}

    validate_clash_rules(attributes, clash_rules)
    return attributes, clash_rules


def select_attributes(attributes, clash_rules):
    def resolve_clashes(attributes, clash_rules):
        def normalize_probabilities(probabilities):
            total = sum(probabilities.values())
            return {key: value / total for key, value in probabilities.items() if value > 0}

        def select_current_attributes(attribute_probs):
            return random.choices(list(attribute_probs.keys()), weights=list(attribute_probs.values()), k=1)[0]

        def apply_clash_rules_bulk(selected_attributes, clash_rules):
            for rule in clash_rules:
                fixed_file = rule["fixed_file"]
                fixed_attributes = rule["fixed_attributes"]
                secondary_file = rule["secondary_file"]
                secondary_attributes = rule["secondary_attributes"]

                if fixed_file in selected_attributes and secondary_file in selected_attributes:
                    fixed_selected = selected_attributes[fixed_file]
                    if fixed_selected in fixed_attributes:
                        secondary_selected = selected_attributes[secondary_file]

                        # Check if the selected attribute in the secondary file is conflicting
                        if secondary_selected in secondary_attributes:
                            attribute_probs = attributes[secondary_file]["attribute_probabilities"]

                            # Remove all conflicting attributes from the secondary file
                            for attr in secondary_attributes:
                                attribute_probs.pop(attr, None)

                            # If no valid attributes remain, remove the file
                            if not attribute_probs:
                                selected_attributes.pop(secondary_file)
                                attributes.pop(secondary_file)
                            else:
                                # Normalize and reselect only if there are valid attributes
                                attribute_probs = normalize_probabilities(attribute_probs)
                                attributes[secondary_file]["attribute_probabilities"] = attribute_probs
                                selected_attributes[secondary_file] = select_current_attributes(attribute_probs)

        random.seed()

        # Step 1: Filter files based on their probability
        filtered_attributes = {
            file: data for file, data in attributes.items() if random.uniform(0, 1) <= data["prob_of_this_file"]
        }

        # Step 2: Initial random selection of attributes
        selected_attributes = {
            file: select_current_attributes(data["attribute_probabilities"])
            for file, data in filtered_attributes.items()
        }

        # Step 3: Iteratively resolve clashes
        while True:
            previous_selection = selected_attributes.copy()
            apply_clash_rules_bulk(selected_attributes, clash_rules)

            if selected_attributes == previous_selection:
                break  # No more clashes to resolve

        return selected_attributes

    attributes = copy.deepcopy(attributes)
    return resolve_clashes(attributes, clash_rules)


def construct_prompt(selected_attributes):
    """Construct a prompt dynamically and position the caption for LLM continuation."""
    p = inflect.engine()
    prompt_parts = []
    gender = None

    for key, value in selected_attributes.items():
        # Remove the file extension and process the filename
        attribute_name = key.replace(".txt", "")
        singular_name = p.singular_noun(attribute_name) or attribute_name  # Convert to singular if plural
        # Format the attribute into a readable string
        formatted_name = singular_name.replace("_", " ").capitalize()

        # Check for gender-related attributes but do not include in the list
        if formatted_name.lower() in ["gender", "genders", "sex", "sexes"]:
            gender = value.lower()
            continue  # Skip adding gender to the input parameter list

        prompt_parts.append(f"{formatted_name}: {value}")

    # Determine caption based on gender
    caption = "s7ll2f4h, Photograph of a "

    if gender in ["male", "man"]:
        subject = "man"
    elif gender in ["female", "woman"]:
        subject = "woman"
    elif gender in ["non-binary"]:
        subject = "non-binary person"
    else:
        # Default to a 50/50 random choice
        subject = "man" if random.choice([True, False]) else "woman"

    caption += f"{subject}. The subject has a neutral face expression."

    # Place caption as the last visible input parameter for LLM continuation
    formatted_parameters = "\n".join(prompt_parts)
    full_prompt = f"{formatted_parameters}\n\nPrompt: {caption}"

    return full_prompt, caption, subject


def load_existing_uuids(dataset_dir):
    return set(f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f)))


def generate_unique_uuid(existing_uuids):
    while True:
        unique_id = uuid.uuid4().hex[:12]  # Generate a 12-character unique ID
        if unique_id not in existing_uuids:
            existing_uuids.add(unique_id)
            return unique_id


def generate_prompts(dataset_dir, number_of_prompts, file_prob_txt, attribute_clash_txt):
    os.makedirs(dataset_dir, exist_ok=True)

    existing_uuids = load_existing_uuids(dataset_dir)

    model_path = "./models/huggingface/Qwen2.5"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    attributes, clash_rules = load_attributes_and_clashes(file_prob_txt, attribute_clash_txt)

    base_prompt = """
        You will be provided with a structured list of input parameters describing details for a photo. 

        Rules for Output:
        1) Combine all provided parameters into one descriptive prompt. Make a short descriptive caption with multiple sentences which explicitly includes every single input parameter.
        2) Make sure to use EVERY SINGLE parameter! DO NOT ignore anything.
        3) DO NOT include unmentioned subjective parameters such as beaty, neatness, tidiness, etc.
        4) All captions will start with: 's7ll2f4h, Photograph of a {man/woman/non-binary person}. The subject has a neutral face expression.'. And your job is to continue it by adding the remaining descriptions.
        5) NEVER start yourself with 's7ll2f4h, Photograph of a {man/woman/non-binary person}. The subject has a neutral face expression.'. Instead, complete the starting prompt directly, without repeating this.
        6) DO NOT include any mentions about the background or clothing.

        Task: Using the given input parameters, complete (start directly where I left off) the given prompt for the diffusion model.
        Do not include explanations, formatting templates, or additional examples — output only the resulting prompt based on the provided data.

        Input Parameters:
    """

    for _ in tqdm(range(number_of_prompts)):
        selected_attributes = select_attributes(attributes, clash_rules)
        dynamic_prompt, caption_start, subject = construct_prompt(selected_attributes)

        final_prompt = base_prompt + dynamic_prompt

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": final_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        unique_uuid = generate_unique_uuid(existing_uuids)
        subfolder_path = os.path.join(dataset_dir, unique_uuid)
        os.makedirs(subfolder_path, exist_ok=True)

        info_file = f"{unique_uuid}_{'m' if subject == 'man' else 'f' if subject == 'woman' else 'nb'}.json"
        info_file_path = os.path.join(subfolder_path, info_file)

        while True:  # Added to avoid unwanted text in captions
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)

            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if caption_start not in generated_text:
                generated_text = (
                    caption_start
                    + (" " if not generated_text.startswith(" ") else "")
                    + generated_text
                    + " Photo captures the subject from the neck up with grey clothing slightly visible around the neck"
                    + ". White background."
                )

            if contains_non_english(generated_text):
                print("Non-English text detected, retrying...")
                continue  # Retry if non-English characters are found

            try:
                # Save attributes and prompt in a single JSON file
                with open(info_file_path, "w", encoding="utf-8") as json_file:
                    json.dump({"attributes": selected_attributes, "prompt": generated_text}, json_file, indent=4)
                break
            except Exception as e:
                print(f"Unexpected error saving file: {e}, retrying...")
                continue  # Retry if there's any unexpected error


if __name__ == "__main__":
    args = parse_args()

    # Call function with parsed arguments
    generate_prompts(args.dataset_dir, args.number_of_prompts, args.file_prob_txt, args.attribute_clash_txt)
