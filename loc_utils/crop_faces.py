"""
The FaceCropper class defined in ths script can be used to automatically crop the innner face regions for use with FRS
models. Three different crop types are available: static, mtcnn_crop and mtcnn_aligned_crop. static simply crops the 
center region of the image. mtcnn_crop crops the face region by first detecting the face using MTCNN model and then 
cropping it and then using adaptive average pooling to achieve the desired crop resolution. mtcnn_aligned_crop detects 
the face using MTCNN model but then also aligns it using reference facial landmarks.
"""

import torch
import torch.nn as nn
from models.mtcnn.mtcnn import MTCNN
import numpy as np
from kornia.geometry.transform import warp_affine
import torchvision.transforms as transforms


class SharedMTCNN:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            instance.mtcnn = MTCNN(device=instance.device, select_largest=True)
            cls._instance = instance
        return cls._instance

    @classmethod
    def get_mtcnn(cls):
        return cls.__new__(cls).mtcnn


class FaceCropper(nn.Module):
    def __init__(self, crop_type="mtcnn_crop", crop_size=112):
        super().__init__()
        assert crop_type in ["static", "mtcnn_crop", "mtcnn_aligned_crop"], "Invalid crop_type provided"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.mtcnn = SharedMTCNN.get_mtcnn()
        self.face_pool = nn.AdaptiveAvgPool2d((crop_size, crop_size))

    def crop(self, x):
        if self.crop_type == "static":
            return self.crop_static(x)
        elif self.crop_type == "mtcnn_crop":
            return self.crop_using_mtcnn(x)
        elif self.crop_type == "mtcnn_aligned_crop":
            return self.crop_and_warp_using_mtcnn(x)

    def detect_num_faces(self, pil_img):
        # Input is a PIL image
        img_tensor = (
            transforms.Compose(
                [
                    transforms.Resize((256, 256), interpolation=1),
                    transforms.ToTensor(),
                ]
            )(pil_img)
            .unsqueeze(0)
            .to(self.device)
        )

        self.mtcnn.select_largest = False
        _, _, landmarks = self.mtcnn.detect((img_tensor * 255), landmarks=True)
        self.mtcnn.select_largest = True

        if landmarks.shape == (1,):
            return 0
        else:
            return landmarks.shape[1]

    def crop_static(self, faces):
        # based on https://github.com/eladrich/pixel2style2pixel/blob/master/criteria/id_loss.py
        y1 = int(35 * faces.shape[2] / 256)
        y2 = int(223 * faces.shape[2] / 256)
        x1 = int(32 * faces.shape[3] / 256)
        x2 = int(220 * faces.shape[3] / 256)
        faces = faces[:, :, y1:y2, x1:x2]  # Crop interesting region
        faces = self.face_pool(faces)
        return faces

    def crop_using_mtcnn(self, faces):
        boxes, _ = self.mtcnn.detect(((faces + 1) * 127.5))

        cropped_faces = []
        for i, box in enumerate(boxes):
            if box is not None:
                # Crop using slicing to keep the operation differentiable
                x1, y1, x2, y2 = [int(b) for b in box[0]]

                box_width = x2 - x1
                box_height = y2 - y1

                # Determine the difference and how much to add to each side to make a square box
                if box_width > box_height:
                    diff = box_width - box_height
                    y1 -= diff // 2
                    y2 += diff // 2
                else:
                    diff = box_height - box_width
                    x1 -= diff // 2
                    x2 += diff // 2

                # Ensure the box does not go out of the image boundaries
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, faces.shape[3])
                y2 = min(y2, faces.shape[2])

                cropped = faces[i, :, y1:y2, x1:x2]
                cropped = self.face_pool(cropped)
                cropped_faces.append(cropped)
            else:
                # If face is not detected, no crop is performed
                # cropped_faces.append(self.face_pool(faces[i, :, :, :]))

                # If face is not detected, static crop is performed
                cropped_faces.append(self.crop_static(faces[i, :, :, :].unsqueeze(0)).squeeze())

        return torch.stack(cropped_faces)

    def crop_and_warp_using_mtcnn(self, faces):
        # Detect face landmarks on using MTCNN
        _, _, landmarks = self.mtcnn.detect(((faces + 1) * 127.5), landmarks=True)

        # Create a mask for valid (non-None) landmarks
        valid_landmarks_mask = [landmark is not None for landmark in landmarks]
        valid_faces = faces[valid_landmarks_mask]

        # Initialize output tensor
        output_img = torch.empty_like(torch.rand((len(faces), faces.shape[1], self.crop_size, self.crop_size))).to(
            self.device
        )

        if len(valid_faces) > 0:
            landmarks_array = np.array([landmark for landmark in landmarks if landmark is not None], dtype=np.float32)
            valid_landmarks = torch.tensor(landmarks_array).squeeze(1).to(self.device)

            # Reference landmarks used with MTCNN in ArcFace for 112x112 images
            ref_points = torch.tensor(
                [
                    [38.29459953, 51.69630051],
                    [73.53179932, 51.50139999],
                    [56.02519989, 71.73660278],
                    [41.54930115, 92.3655014],
                    [70.72990036, 92.20410156],
                ]
            ).to(self.device)
            ref_points_batch = ref_points.unsqueeze(0).repeat(valid_landmarks.shape[0], 1, 1)

            # Estimate affine transformation
            ones = torch.ones((valid_landmarks.shape[0], valid_landmarks.shape[1], 1), device=self.device)
            src_aug = torch.cat((valid_landmarks, ones), dim=2)
            result = torch.linalg.lstsq(src_aug, ref_points_batch)
            M_torch_batch = result.solution[:, :3].transpose(1, 2)

            # Apply affine transformation to align the face
            output_img[valid_landmarks_mask] = warp_affine(valid_faces, M_torch_batch, (self.crop_size, self.crop_size))

        if len(faces) - len(valid_faces) > 0:
            # For faces with None landmarks, apply identity transformation
            identity_matrix = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32).to(self.device)
            identity_matrix_batch = identity_matrix.unsqueeze(0).repeat(len(faces) - len(valid_faces), 1, 1)

            output_img[~torch.tensor(valid_landmarks_mask)] = warp_affine(
                faces[~torch.tensor(valid_landmarks_mask)], identity_matrix_batch, (self.crop_size, self.crop_size)
            )
        return output_img
