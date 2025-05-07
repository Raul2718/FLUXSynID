#!/bin/bash

# Fail early if something goes wrong
set -e

# Step 1: Detect GPU compute capability
echo "Detecting GPU compute capability..."

# Check if nvidia-smi supports querying compute capability
if nvidia-smi --help-query-gpu | grep -q 'compute_cap'; then
    compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
else
    echo "Your version of nvidia-smi doesn't support querying compute capability directly. Defaulting to compute capability of 8.0."
    compute_capability="8.0"
fi

arch="${compute_capability}+PTX"
echo "Using TORCH_CUDA_ARCH_LIST=\"$arch\""

# Step 2: Update the Dockerfile
dockerfile="Dockerfile"

if grep -q "^ENV TORCH_CUDA_ARCH_LIST=" "$dockerfile"; then
    # Replace the line
    sed -i "s/^ENV TORCH_CUDA_ARCH_LIST=.*/ENV TORCH_CUDA_ARCH_LIST=\"$arch\"/" "$dockerfile"
else
    echo "Failed to find TORCH_CUDA_ARCH_LIST variable in Dockerfile"
fi

# Step 3: Build Docker image
image_name="fluxsynid"
echo "Building Docker image: $image_name"

docker build -t "$image_name" .

echo "Done! Image built with CUDA arch: $arch"
