import os
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

def download_file(url, destination):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {destination.name}")
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
        return False
    return True

def extract_zip(zip_path, extract_to):
    """Extracts a zip file with basic logging."""
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def setup_imfdb():
    # Configuration
    # Using the direct CDN link for stability
    url = "https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/projects/IMFDB/db.zip"
    raw_data_dir = Path("data/raw/imfdb")
    zip_path = raw_data_dir / "imfdb_images.zip"
    
    # Create directory structure
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- IMFDB Downloader ---")
    print(f"Target Directory: {raw_data_dir}")
    
    # Download
    if not zip_path.exists():
        success = download_file(url, zip_path)
        if not success:
            print("Download failed.")
            return
    else:
        print(f"Zip file already exists at {zip_path}. Skipping download.")
    
    # Extract
    # We check for a sample folder to see if it's already extracted
    # IMFDB usually contains folders per actor (e.g., 'AamirKhan')
    if zip_path.exists():
        extract_zip(zip_path, raw_data_dir)
        
        # Cleanup zip to save space if desired, but keeping it for now for safety
        # os.remove(zip_path)
        
    print("\nSUCCESS: IMFDB dataset is ready in data/raw/imfdb")
    print(f"Total actors found: {len([d for d in raw_data_dir.iterdir() if d.is_dir()])}")

if __name__ == "__main__":
    setup_imfdb()
