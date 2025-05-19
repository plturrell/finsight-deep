#!/bin/bash

# Compile the protocol buffer definitions for NVIDIA Audio2Face

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROTO_DIR="${SCRIPT_DIR}/protos"
OUT_DIR="${SCRIPT_DIR}/generated"

# Create output directory
mkdir -p "${OUT_DIR}"

# Compile proto files
python -m grpc_tools.protoc \
    -I="${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/audio2face.proto"

# Create __init__.py file
touch "${OUT_DIR}/__init__.py"

echo "Proto files compiled successfully!"
echo "Generated files in: ${OUT_DIR}"