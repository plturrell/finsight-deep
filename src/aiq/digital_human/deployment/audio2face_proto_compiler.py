#!/usr/bin/env python
"""Compile the Audio2Face protocol buffer definitions"""

import subprocess
import sys
import os
from pathlib import Path


def compile_protos():
    """Compile protocol buffer files for NVIDIA Audio2Face"""
    # Get paths
    script_dir = Path(__file__).parent
    proto_dir = script_dir / "protos"
    out_dir = script_dir / "generated"
    
    # Create output directory
    out_dir.mkdir(exist_ok=True)
    
    # Find proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print(f"No proto files found in {proto_dir}")
        return False
    
    # Compile each proto file
    for proto_file in proto_files:
        print(f"Compiling {proto_file.name}...")
        
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I={proto_dir}",
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            str(proto_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error compiling {proto_file.name}:")
                print(result.stderr)
                return False
            else:
                print(f"Successfully compiled {proto_file.name}")
        except Exception as e:
            print(f"Failed to compile {proto_file.name}: {e}")
            return False
    
    # Create __init__.py
    init_file = out_dir / "__init__.py"
    init_file.touch()
    
    print(f"\nProto files compiled successfully!")
    print(f"Generated files in: {out_dir}")
    return True


if __name__ == "__main__":
    # Install required packages first
    print("Installing grpcio-tools if needed...")
    subprocess.run([sys.executable, "-m", "pip", "install", "grpcio-tools"], 
                   capture_output=True)
    
    # Compile protos
    success = compile_protos()
    sys.exit(0 if success else 1)
