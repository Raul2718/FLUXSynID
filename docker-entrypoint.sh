#!/bin/bash

# Base project directory
BASE_DIR="/FLUXSynID"

# Arc2Face paths
ARC2FACE_DIR="${BASE_DIR}/Arc2Face/models"
ARC2FACE_LINK="${ARC2FACE_DIR}/antelopev2"
ARC2FACE_TARGET="${BASE_DIR}/models/Arc2Face/antelopev2"

# ComfyUI paths
COMFYUI_DIR="${BASE_DIR}/ComfyUI/models/insightface"
COMFYUI_LINK="${COMFYUI_DIR}/models"
COMFYUI_TARGET="${BASE_DIR}/models/ComfyUI/insightface/models"

# Ensure parent dirs exist
mkdir -p "$ARC2FACE_DIR"
mkdir -p "$COMFYUI_DIR"

# Ensure ComfyUI/output/exp_data exists
EXP_DATA_DIR="/FLUXSynID/ComfyUI/output/exp_data"
mkdir -p "$EXP_DATA_DIR"
echo "Ensured directory exists: $EXP_DATA_DIR"

# Create Arc2Face symlink if it doesn't exist
if [ ! -L "$ARC2FACE_LINK" ] && [ -d "$ARC2FACE_TARGET" ]; then
  echo "Creating symlink: $ARC2FACE_LINK -> $ARC2FACE_TARGET"
  ln -s "$ARC2FACE_TARGET" "$ARC2FACE_LINK"
fi

# Create ComfyUI symlink if it doesn't exist
if [ ! -L "$COMFYUI_LINK" ] && [ -d "$COMFYUI_TARGET" ]; then
  echo "Creating symlink: $COMFYUI_LINK -> $COMFYUI_TARGET"
  ln -s "$COMFYUI_TARGET" "$COMFYUI_LINK"
fi

# Run user command or default to bash
if [ $# -eq 0 ]; then
  exec /bin/bash
else
  exec "$@"
fi
