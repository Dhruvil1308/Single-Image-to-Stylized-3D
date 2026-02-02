import os
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import concurrent.futures

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocess.extractor import FaceExtractor

def process_actor_folder(actor_name, raw_dir, proc_dir, extractor, limit=None):
    """Processes all images for a single actor."""
    actor_raw_path = raw_dir / actor_name
    actor_proc_path = proc_dir / actor_name
    actor_proc_path.mkdir(parents=True, exist_ok=True)
    
    images = list(actor_raw_path.glob("*.jpg"))
    if limit:
        images = images[:limit]
        
    count = 0
    for img_path in images:
        out_path = actor_proc_path / img_path.name
        if out_path.exists():
            count += 1
            continue
            
        try:
            aligned_img = extractor.align_and_crop(str(img_path))
            if aligned_img:
                aligned_img.save(out_path)
                count += 1
        except Exception as e:
            # print(f"Error processing {img_path}: {e}")
            pass
    return count

def run_preprocessing(limit_per_actor=None):
    raw_dir = Path("data/raw/imfdb")
    proc_dir = Path("data/processed/imfdb")
    
    if not raw_dir.exists():
        print(f"Error: Raw data directory {raw_dir} not found.")
        return

    extractor = FaceExtractor()
    actors = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    
    print(f"--- IMFDB Preprocessor ---")
    print(f"Actors found: {len(actors)}")
    print(f"Output Directory: {proc_dir}")
    
    total_processed = 0
    
    # Using ThreadPoolExecutor for faster IO/CPU mix, 
    # though MediaPipe might have internal threading/GPU usage
    with tqdm(total=len(actors), desc="Processing actors") as pbar:
        for actor in actors:
            count = process_actor_folder(actor, raw_dir, proc_dir, extractor, limit=limit_per_actor)
            total_processed += count
            pbar.update(1)
            
    print(f"\nSUCCESS: Preprocessing complete!")
    print(f"Total aligned faces saved: {total_processed}")

if __name__ == "__main__":
    # For initial testing, you might want to limit to 10 images per actor
    # Run with: python scripts/preprocess_imfdb.py
    run_preprocessing(limit_per_actor=None)
