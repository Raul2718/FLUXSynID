# FLUXSynID

**FLUXSynID** is a framework for generating high-resolution synthetic face datasets with controllable identity attributes. It produces paired document-style and live capture images per identity, making it ideal for biometric research tasks such as face recognition and morphing attack detection.

![FLUXSynID Example Images](assets/examples.jpg)

## Table of Contents

- [Installing](#installing)
  - [Linux/Windows/WSL (Shared Steps)](#linuxwindowswsl-shared-steps)
  - [Final Setup (Per OS)](#final-setup-per-os)
    - [Linux/WSL-only](#linuxwsl-only)
    - [Windows-only](#windows-only)
  - [Docker](#docker)
- [Dataset Generation](#dataset-generation)
  - [Setting Identity Attributes](#setting-identity-attributes)
  - [Generating Prompts](#generating-prompts)
  - [Generating Document-Style Images](#generating-document-style-images)
  - [Generating Live Capture Images](#generating-live-capture-images)
- [Similarity-Based Identity Filtering](#similarity-based-identity-filtering)
  - [Face Recognition Model Setup](#face-recognition-model-setup)
  - [Deleting Similar Identities](#deleting-similar-identities)

## Installing

The framework has been validated using Python 3.11. Both Linux and Windows OS are supported, with Docker image being the easiest to run.

### Linux/Windows/WSL (Shared Steps)

Follow these common steps regardless of your OS:

- Git clone this repo:

  ```bash
  git clone https://github.com/Raul2718/FLUXSynID.git
  ```

- Set up a virtual Python environment:

  ```bash
  python -m venv .venv
  ```

- Activate your virtual environment:
  - On Linux/WSL:
    ```bash
    source .venv/bin/activate
    ```
  - On Windows (CMD or PowerShell):
    ```powershell
    .venv\Scripts\activate
    ```

- Install PyTorch 2.7 with CUDA 12.8 support:

  ```bash
  pip install torch==2.7.0+cu128 \
      torchvision==0.22.0+cu128 \
      torchaudio==2.7.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128
  ```

- Install `wheel`:

  ```bash
  pip install wheel==0.45.1
  ```

- Install GPTQModel:

  ```bash
  pip install --no-build-isolation -v --no-cache-dir gptqmodel==2.2.0
  ```

- Install packages from `requirements.txt`:

  ```bash
  pip install --no-cache-dir -r requirements.txt
  ```

- Download HuggingFace models:

  1. Go to [https://huggingface.co/settings/tokens/new](https://huggingface.co/settings/tokens/new)
  2. Set a token name (e.g., `FLUX`)
  3. Tick all user permissions
  4. In **Repositories permissions**, enter `black-forest-labs/FLUX.1-dev`
  5. The page should look like this:

     ![HF Token](assets/HF_token.png)

  6. Create the token and **copy it**
  7. Paste it into the `download_models.py` file on line 7:

     ```python
     HF_TOKEN = "PUT_YOUR_TOKEN_HERE"
     ```

  8. Download all models:

     ```bash
     python download_models.py
     ```

  9. Ensure all models are downloaded without errors

### Final Setup (Per OS)

Depending on your operating system, follow the appropriate final setup step:

#### Linux/WSL-only

- Run the setup script:

  ```bash
  bash setup.sh
  ```

#### Windows-only

- Install `triton-windows`:

  ```bash
  pip install triton-windows==3.3.0.post19
  ```

- Run the setup script:

  ```bash
  python win_setup.py
  ```

- If your machine has multiple GPUs, ensure only one is visible to the system:

  ```powershell
  $env:CUDA_VISIBLE_DEVICES="0"
  ```

  > **Note:** This command is for PowerShell. In Bash, use:
  > `export CUDA_VISIBLE_DEVICES=0`

### Docker
WIP...

## Dataset Generation

### Setting Identity Attributes

The `./attributes/attributes` folder contains 14 pre-defined classes of attributes (e.g., `ages.txt`) which define options and their probabilities. Two configuration files further control how attributes are applied:

- `file_probabilities.json`: sets how likely each attribute class (file) is used in a prompt.
- `attribute_clashes.json`: defines incompatible attribute combinations. When a clash occurs, the fixed attribute is retained.

To modify or extend these attributes, run the configuration GUI:

```bash
python -m attributes.prob_settings_app
```

This opens a GUI with several functions:

#### Main Screen

![Main screen](assets/attributes_main_screen.png)

- **Create New File** (left-click): Add a new attribute class (e.g., `eye_color.txt`). *Use meaningful, descriptive names with underscores* (e.g., `hair_type`, `eye_color`). These names are important because a language model will infer the meaning of the attribute class based on the filename.

- **Edit File** (click existing file): Open a text editor to define the values of that class. Each line should contain one attribute (e.g., `Brown`, `Blue` for eye color). Save using the provided button and return to the main screen.

- **Edit Attribute Probabilities** (right-click a file):

  ![Attribute probabilities](assets/attributes_attribute_probs.png)

  - Set how likely each attribute in the file should appear. Probabilities must sum to 100%.

- **Delete File** (right-click): Remove an attribute class entirely.

#### File Usage Probability

Click `Edit Probability for Each Text File` on the main screen to open:

![File probabilities](assets/attributes_file_probs.png)

- Adjust how often each attribute class is used in identity prompts.
- For example, age might be used in 100% of identities, while body type might only be used 50% of the time.
- These settings are saved in `file_probabilities.json`.

#### Attribute Clashes

Click `Declare Attribute Clashes` on the main screen to open:

![Clashes](assets/attributes_clashes.png)

- Select a fixed attribute file (e.g., `hair_styles.txt`) and a secondary file (e.g., `hair_colors.txt`).
- Choose attributes from each that are incompatible (e.g., `Bald` with `Black`).
- When a fixed attribute is selected during prompt generation, conflicting secondary attributes are removed.
- Clashes are saved in `attribute_clashes.json`.

### Generating Prompts

Generate identity prompts based on your attributes:

```bash
python -m scripts.generate_prompts --dataset_dir FLUXSynID --num 15000
```

This creates a folder `FLUXSynID` with 15,000 subfolders, each defining one identity. Adjust `--dataset_dir` and `--num` as needed.

### Generating Document-Style Images

To generate document-style images:

```bash
python -m scripts.generate_document_imgs --dataset_dir FLUXSynID
```

Each identity will receive one generated document-style image.

### Generating Live Capture Images

To generate **LivePortrait** and **PuLID** live images:

```bash
python -m scripts.generate_live_imgs --dataset_dir FLUXSynID --num_live_imgs 1
```

To generate **Arc2Face** live images:

```bash
python -m scripts.generate_live_imgs_arc2face --dataset_dir FLUXSynID --num_live_imgs 1
```

You can increase the number of live images per identity using the `--num_live_imgs` flag.

## Similarity-Based Identity Filtering

Optionally, and *preferably before live image generation (right after document image generation)*, you can remove identities which are too visually similar to each other. This ensures dataset diversity and reduces duplicate-like samples.

To run similarity filtering:

```bash
python -m dataset_filtering.find_similar_ids.py --dataset_dir FLUXSynID --frs adaface --fmr 0.0001
```

This generates a text file:

```bash
./dataset_filtering/similarity_filtering_adaface_thr_0.333987832069397_fmr_0.0001.txt
```

The file lists all identity folders that contain diverse identities and should be kept. Currently supported face recognition systems (FRS): **AdaFace**, **ArcFace**, and **CurricularFace**.

- Use the `--frs` flag to choose the FRS model.
- Use the `--fmr` flag to define the False Match Rate (supports `0.001` and `0.0001`).

### Face Recognition Model Setup

Before running the script, place the weights of the chosen FRS model in the appropriate directory:

- **AdaFace**: [Google Drive link](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view)
  - Save to: `.models/face_recognition/adaface/checkpoint`

- **ArcFace**: [OneDrive link](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC)
  - Rename file to `ArcFace.pth`
  - Save to: `.models/face_recognition/arcface/checkpoint`

- **CurricularFace**: [Google Drive link](https://drive.google.com/open?id=1upOyrPzZ5OI3p6WkA5D5JFYCeiZuaPcp)
  - Rename file to `CurricularFace.pth`
  - Save to: `.models/face_recognition/curricularface/checkpoint`

### Deleting Similar Identities

Once the similarity filtering script has generated the list, you can use it to retain only the listed identities.

To apply the filtering and delete all other folders from your dataset:

```bash
python -m dataset_filtering.delete_present_folders --dataset_path FLUXSynID --txt_path <PATH_TO_TXT_GENERATED_BEFORE>
```
