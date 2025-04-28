# Use NVIDIA base image with CUDA 12.8
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set to match your GPU compute capability (e.g. 8.9+PTX). See build_image.sh for auto-detect.
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"

ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /FLUXSynID

# Install system dependencies, Python 3.11, and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    lsb-release \
    ca-certificates \
    nano \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    python3-tk \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libstdc++6 \
    gcc \
    g++ \
    llvm \
    clang \
    zlib1g-dev && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8
RUN pip install \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install wheel
RUN pip install wheel==0.45.1

# Install gptqmodel
RUN pip install --no-build-isolation -v --no-cache-dir gptqmodel==2.2.0

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

RUN rm requirements.txt

# Set the default entrypoint
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
