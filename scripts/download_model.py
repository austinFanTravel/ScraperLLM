#!/usr/bin/env python3
"""Download the Mistral 7B model."""
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, filename: str):
    """Download a file with progress bar."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {filename}...")
    
    # Stream the download to show progress
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size from headers
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB chunks
    
    with open(filename, 'wb') as f, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)
    
    print(f"\nDownload complete: {filename}")

def main():
    """Main function to download the model."""
    # Model information
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    model_path = model_dir / "mistral-7b-instruct.Q4_K_M.gguf"
    
    try:
        download_file(model_url, str(model_path))
        print("\nModel downloaded successfully!")
    except Exception as e:
        print(f"\nError downloading model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
