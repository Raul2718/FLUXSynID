import torch
import torch.nn.functional as F
import numpy as np
import time
from models.face_recognition.adaface.model import AdaFace
from models.face_recognition.arcface.model import ArcFace
from models.face_recognition.curricularface.model import CurricularFace
from loc_utils.crop_faces import FaceCropper
from torchvision import transforms
import torch.nn.functional as F
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

THRESHOLDS = {
    "arcface": {0.001: 0.42314568161964417, 0.0001: 0.49696439504623413},
    "adaface": {0.001: 0.2530066967010498, 0.0001: 0.333987832069397},
    "curricularface": {0.001: 0.2855643630027771, 0.0001: 0.3750773072242737},
    "arcface-onot": {0.001: 0.5068290536891065, 0.0001: 0.5868447527510883},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Filter out similar identities.")

    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument(
        "--frs",
        type=str,
        required=True,
        choices=["arcface", "adaface", "curricularface", "arcface-onot"],
        help="FRS used for filtering.",
    )
    parser.add_argument("--fmr", type=float, required=True, choices=[0.001, 0.0001], help="FMR value for the dataset.")

    args = parser.parse_args()

    return args


class FaceDataset(Dataset):
    def __init__(self, dataset_dir, transform, allowed_folders=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.samples = []

        for folder in os.listdir(dataset_dir):
            if allowed_folders is not None and folder not in allowed_folders:
                continue  # Skip folders not in the list

            folder_path = os.path.join(dataset_dir, folder)
            doc_img_files = [f for f in os.listdir(folder_path) if f.endswith("_doc.png")]

            if len(doc_img_files) == 1:
                self.samples.append((folder, os.path.join(folder_path, doc_img_files[0])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, img_path = self.samples[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return folder, img


def generate_features_dict(dataset_dir, frs, folder_list_path=None, batch_size=64, num_workers=16):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    # Load allowed folders if provided
    allowed_folders = None
    if folder_list_path:
        if not os.path.exists(folder_list_path):
            raise FileNotFoundError(f"Folder list file '{folder_list_path}' does not exist.")

        with open(folder_list_path, "r") as f:
            allowed_folders = {line.strip() for line in f if line.strip()}

    face_cropper = FaceCropper(crop_type="mtcnn_aligned_crop")

    if frs in ["arcface", "arcface-onot"]:
        frs = ArcFace()
    elif frs == "adaface":
        frs = AdaFace()
    elif frs == "curricularface":
        frs = CurricularFace()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = FaceDataset(dataset_dir, transform, allowed_folders)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features_dict = {}

    for batch in tqdm(dataloader):
        folders, images = batch
        images = images.to(face_cropper.device)

        with torch.no_grad():
            cropped_faces = face_cropper.crop(images)  # Batch crop
            features = frs(cropped_faces)  # Batch feature extraction

        for folder, feature in zip(folders, features):
            features_dict[folder] = feature.cpu().squeeze()

    return features_dict


def greedy_mis_filter_optimized(features_dict, frs, target_fmr):
    """
    Optimized greedy algorithm to filter similar identities such that
    the overall false match rate (FMR) among the remaining identities is at most target_fmr.
    Fully runs on GPU using PyTorch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold = THRESHOLDS[frs][target_fmr]

    keys = list(features_dict.keys())
    features = [features_dict[k] for k in keys]
    X = torch.stack(features).to(device)  # Shape: (n, FEATURE_DIM)
    X_norm = F.normalize(X, p=2, dim=1)

    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(X_norm, X_norm.t())
    S = (similarity_matrix >= threshold).int()  # Binary adjacency matrix

    n = S.shape[0]
    S[torch.arange(n, device=device), torch.arange(n, device=device)] = 0  # Remove self-edges

    degrees = S.sum(dim=1)  # Compute degrees
    active = torch.ones(n, dtype=torch.bool, device=device)  # Active nodes

    while True:
        active_indices = torch.nonzero(active, as_tuple=True)[0]  # Get active indices
        M = active_indices.numel()
        if M <= 1:
            break

        # Compute current FMR
        submatrix = S[active_indices][:, active_indices]  # Get submatrix of active nodes
        total_edges = submatrix.sum().item()
        total_possible = M * (M - 1)
        current_fmr = total_edges / total_possible

        if frs == "arcface-onot" and current_fmr == 0.0:
            break
        elif frs != "arcface-onot" and current_fmr <= target_fmr:
            break

        # Find the node with the highest degree
        max_degree, node_to_remove_idx = torch.max(degrees[active_indices], dim=0)
        node_to_remove = active_indices[node_to_remove_idx]

        # Remove node
        active[node_to_remove] = False

        # Update degrees for remaining nodes
        neighbors = torch.nonzero(S[node_to_remove], as_tuple=True)[0]
        S[node_to_remove, :] = 0  # Remove edges from node
        S[:, node_to_remove] = 0  # Remove edges to node

        for j in neighbors:
            if active[j]:
                degrees[j] -= 1  # Reduce degree count

        degrees[node_to_remove] = 0  # Clear removed node's degree

    # Remaining active nodes form the filtered dataset
    remaining_indices = torch.nonzero(active, as_tuple=True)[0]
    remaining_keys = [keys[i] for i in remaining_indices.tolist()]
    filtered_dict = {k: features_dict[k] for k in remaining_keys}

    return filtered_dict, remaining_keys


if __name__ == "__main__":
    args = parse_args()

    # folder_list_path is a path to txt file that contains folders (ids) which should be considered by this script
    # useful when filtering identities previously filtered by ICAO compliance
    folder_list_path = None

    features_dict = generate_features_dict(
        dataset_dir=args.dataset_dir, frs=args.frs, folder_list_path=folder_list_path
    )

    # Time the filtering process.
    start_time = time.time()
    filtered_dict, remaining_keys = greedy_mis_filter_optimized(features_dict, frs=args.frs, target_fmr=args.fmr)
    end_time = time.time()

    print("Original number of identities:", len(features_dict))
    print("Remaining number of identities:", len(filtered_dict))
    print("Time taken: {:.4f} seconds".format(end_time - start_time))

    # Save the kept keys to a text file.
    with open(
        f"dataset_filtering/similarity_filtering_{args.frs}_thr_{THRESHOLDS[args.frs][args.fmr]}_fmr_{args.fmr}.txt",
        "w",
    ) as f:
        for key in remaining_keys:
            f.write(os.path.basename(key) + "\n")

    print(
        f"All results have been written to dataset_filtering/similarity_filtering_{args.frs}_thr_{THRESHOLDS[args.frs][args.fmr]}_fmr_{args.fmr}.txt"
    )
