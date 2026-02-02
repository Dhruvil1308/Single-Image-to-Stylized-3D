# 🇮🇳 Indian Avatar AI: 2D & 3D Stylization System

## 🌟 Introduction
The **Indian Avatar AI** is a state-of-the-art generative system designed to transform standard facial photographs into culturally accurate, high-quality Indian avatars. The project supports two primary output paths:
1.  **2D Anime Stylization**: Leveraging Stable Diffusion with specialized LoRA fine-tuning for Indian aesthetics.
2.  **3D Avatar Reconstruction**: Utilizing FLAME morphable models and multi-view synthesis to create realistic 3D head geometry.

By focusing specifically on Indian facial structures, skin tones, and traditional elements, this project aims to provide a more representative and personalized avatar experience compared to generic global models.

---

## 🛠️ Tech Stack & Libraries
The project is built using a modern AI/ML stack, optimized for efficiency and local execution:

*   **Core Logic**: Python 3.11
*   **Deep Learning**: PyTorch, HuggingFace Diffusers, Transformers, PEFT (Parameter-Efficient Fine-Tuning).
*   **Computer Vision**: MediaPipe (Face Detection/Mesh), OpenCV, PIL (Pillow).
*   **3D Modeling**: FLAME (Face Learned Model Entities), 3DMM fitting algorithms.
*   **Interface**: Gradio (Web UI).
*   **Optimizations**: BitsAndBytes (8-bit quantization), Accelerate, Gradient Checkpointing.

---

## 🤖 Models & Algorithms

### 1. 2D Generation: Stable Diffusion + LoRA
*   **Base Model**: `Lykon/AnyLoRA` (highly flexible SD1.5-based model).
*   **Fine-tuning (LoRA)**: A custom LoRA trained on the IMFDB dataset with the trigger word `indianface`. It captures features like `bindis`, `saris`, `dhotis`, and diverse Indian skin tones.
*   **Identity Preservation**: Uses IP-Adapter/FaceID techniques to ensure the avatar looks like the original user.

### 2. 3D Path: Morphable Diffusion & FLAME
*   **Morphable Diffusion**: A framework that generates consistent multi-view images (Front, Side, Quarter) from a single photo.
*   **FLAME Wrapper**: We use the FLAME morphable model to represent 3D head geometry with thousands of trainable parameters for shape and expression.
*   **Mesh Fitting**: A custom iterative algorithm that fits the FLAME mesh to the detected landmarks from the generated multi-view images.

---

## 📊 Dataset: IMFDB (Indian Movie Face Database)
The heart of our model's "Indian intelligence" is the **IMFDB dataset**.
*   **Source**: IIIT Hyderabad Academic CDN.
*   **Scale**: 34,512 images of 100 Indian actors.
*   **Our Processing**:
    *   **Alignment**: Every image is automatically rotated to level the eyes.
    *   **Cropping**: Tight, centralized face crops (512x512) optimized for training.
    *   **Cleaned Set**: We maintain a processed set of ~25,000 high-quality portraits for fine-tuning.

*Note: The **IFExD** dataset was initially considered but confirmed as "TBD" (To Be Determined) by its authors, leading us to successfully pivot to IMFDB.*

---

## 🔄 Workflow
1.  **Upload**: User uploads a front-facing photo via the Gradio interface.
2.  **Preprocessing**:
    *   `FaceExtractor` detects landmarks and aligns the face.
    *   `Segmenter` removes the background.
3.  **Routing**:
    *   **2D Path**: The image is passed to `AnimeGenerator` with a custom style prompt (Realistic, Cartoon, or Stylized) + the `indianface` LoRA.
    *   **3D Path**: `MorphableDiffusion` generates 7 views -> `MeshFitter` creates 3D geometry -> `Exporter` saves as `.obj`.
4.  **Result**: The user receives a high-resolution avatar or a 3D model preview.

---

## ⚠️ Challenges & Solutions

### 1. 📉 VRAM Constraints (4GB Limit)
**Challenge**: Training and running Large Language/Diffusion models usually requires 8GB-24GB VRAM.
**Solution**:
*   Implemented **8-bit AdamW Optimization** via `bitsandbytes`.
*   Enabled **Gradient Checkpointing** to trade compute time for memory savings.
*   Used **Sequential CPU Offloading** in the inference pipeline to only keep active layers in VRAM.

### 2. 🧩 Windows Compatibility (xformers DLL Error)
**Challenge**: `xformers` (a memory optimization library) often fails to load its C++/CUDA extensions on Windows.
**Solution**: We transitioned to **Standard PyTorch Attention** with `enable_gradient_checkpointing`. We also performed a clean uninstallation of broken `xformers` builds to ensure stable execution.

### 3. 🌐 Large Data Protocol Errors
**Challenge**: Gradio/Uvicorn threw `Too much data for declared Content-Length` errors when transferring high-res images.
**Solution**: Tuned the web server configuration using `max_file_size="10mb"` and optimized the image quantization before transmission.

### 4. 🔗 Dependency Hell (MediaPipe vs Protobuf)
**Challenge**: New versions of MediaPipe had breaking changes regarding the `solutions` attribute and `protobuf` versioning.
**Solution**: Pinned `mediapipe==0.10.14` and ensured `pydantic` and `fastapi` versions were synchronized for the Gradio 5+ environment.

---

## 🖥️ Hardware Requirements
*   **GPU**: NVIDIA 3050 4GB (Minimum). Compatible with other 4GB+ NVIDIA cards.
*   **RAM**: 16GB (Recommended).
*   **Storage**: 5GB for models + 50GB for raw datasets.

---

## 🚀 Getting Started
1. Initialize environment: `python scripts/setup_venv.py`
2. Download Data: `python scripts/download_imfdb.py`
3. Preprocess: `python scripts/preprocess_imfdb.py`
4. Train (Optional): `python scripts/train_lora.py`
5. Run App: `python app.py`

Developed by **Our GUNI Research Team**
