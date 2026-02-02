import os
import json
from pathlib import Path

def prepare_metadata():
    proc_dir = Path("data/processed/imfdb")
    metadata_file = proc_dir / "metadata.jsonl"
    
    if not proc_dir.exists():
        print(f"Error: Processed directory {proc_dir} not found.")
        return

    print("--- Preparing LoRA Metadata ---")
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        # We iterate through each actor folder
        actors = [d for d in proc_dir.iterdir() if d.is_dir()]
        
        for actor_dir in actors:
            actor_name = actor_dir.name
            # Simplified caption for LoRA: trigger word 'indianface' + descriptive context
            images = list(actor_dir.glob("*.jpg"))
            
            for img_path in images:
                # Path relative to the metadata file
                rel_path = f"{actor_name}/{img_path.name}"
                
                # Metadata entry
                entry = {
                    "file_name": rel_path,
                    "text": f"a professional portrait of an indian person, {actor_name}, indianface style, high quality"
                }
                
                f.write(json.dumps(entry) + "\n")
                
    print(f"SUCCESS: Metadata saved to {metadata_file}")
    print(f"Trigger word: 'indianface'")

if __name__ == "__main__":
    prepare_metadata()
