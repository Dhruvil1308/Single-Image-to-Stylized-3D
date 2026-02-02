import os
import subprocess
import sys
import shutil

def create_venv():
    venv_dir = "venv"
    if os.path.exists(venv_dir):
        print(f"Virtual environment '{venv_dir}' already exists. Deleting to recreate with Python 3.11...")
        shutil.rmtree(venv_dir)

    print("Checking for Python 3.11...")
    try:
        # Check if py launcher can find 3.11
        subprocess.run(["py", "-3.11", "--version"], check=True, capture_output=True)
        python_cmd = ["py", "-3.11"]
        print("Python 3.11 found. Creating virtual environment...")
    except Exception:
        print("Python 3.11 NOT found via 'py -3.11'.")
        print("Please download and install Python 3.11 from python.org first.")
        return

    subprocess.run(python_cmd + ["-m", "venv", venv_dir], check=True)
    print("Virtual environment created successfully with Python 3.11.")

    # Provide activation instructions
    if os.name == "nt": # Windows
        activate_script = os.path.join(venv_dir, "Scripts", "activate")
        print(f"\nTo activate the environment, run:\n    {activate_script}")
    else: # Unix/macOS
        activate_script = os.path.join(venv_dir, "bin", "activate")
        print(f"\nTo activate the environment, run:\n    source {activate_script}")

    print("\nAfter activation, install dependencies with:\n    pip install -r requirements.txt")

if __name__ == "__main__":
    create_venv()
