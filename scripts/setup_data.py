import os

def setup_directories():
    base_dir = "d:/3d_model"
    dirs = [
        "data/raw/imfdb",
        "data/raw/ifexd",
        "data/raw/kaggle_indian_face",
        "data/processed/aligned",
        "assets/models",
        "assets/exports"
    ]
    for d in dirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def print_dataset_info():
    print("\n--- Dataset Setup Instructions ---")
    print("To make the project 'ready', please download the following datasets as per the project requirements:")
    
    datasets = {
        "IMFDB": "https://www.kaggle.com/datasets/anirudhsimhachalam/indian-movie-faces-datasetimfdb-face-recognition",
        "IFExD": "https://ifexd.github.io/dataset",
        "Kaggle Indian Face": "https://www.kaggle.com/datasets/aryankashyapnaveen/indian-face-dataset",
        "Roboflow Indian Face": "https://universe.roboflow.com/face2/face-indian"
    }
    
    for name, url in datasets.items():
        print(f"\n{name}:")
        print(f"  URL: {url}")
        print(f"  Target Path: d:/3d_model/data/raw/{name.lower().replace(' ', '_')}")

    print("\nOnce downloaded, place the images/folders into their respective target paths.")
    print("The system will automatically detect them for fine-tuning or feature detection.")

if __name__ == "__main__":
    setup_directories()
    print_dataset_info()
