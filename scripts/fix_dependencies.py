import subprocess
import sys
import os

def install_insightface_fix():
    print("\nAttempting to install 'insightface' with pre-compiled wheel for Python 3.11...")
    wheel_url = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl"
    wheel_name = "insightface-0.7.3-cp311-cp311-win_amd64.whl"
    
    try:
        # Check if already installed
        import insightface
        print("'insightface' is already installed.")
        return True
    except ImportError:
        pass

    try:
        print(f"Downloading pre-compiled wheel from {wheel_url}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
        print("'insightface' installed successfully via wheel.")
        return True
    except Exception as e:
        print(f"Wheel installation failed: {e}")
        print("Please ensure you have internet access.")
        return False

def install_chumpy_fix():
    print("\nAttempting to install 'chumpy' directly from GitHub...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/mattloper/chumpy.git"])
        print("'chumpy' installed successfully.")
        return True
    except Exception as e:
        print(f"Chumpy installation failed: {e}")
        return False

if __name__ == "__main__":
    if install_insightface_fix():
        install_chumpy_fix()
    else:
        print("Installation process had some issues. Check the errors above.")
