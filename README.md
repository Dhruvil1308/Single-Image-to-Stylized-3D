# 🇮🇳 Indian Avatar AI: Single-Image to High-Fidelity 3D (v2.0)

> **"Bridging the Representation Gap in Generative AI: A High-Fidelity 3D Avatar Framework for the Indian Demographic."**

---

## 💎 Project Abstract
Traditional 3D face reconstruction and generative AI pipelines often struggle with **Indian facial morphology**, diverse skin tones, and regional aesthetics due to Western-centric training data. The **Indian Avatar AI** project introduces a robust, research-grade pipeline designed to generate smooth, animatable, and culturally accurate 3D avatars from a single unconstrained face photo.

The framework utilizes a combination of **Parametric 3DMM Regression**, **ControlNet-guided View Synthesis**, and the **Perfection v2.0 Mesh Refining Engine** to output high-resolution professional avatars ready for metaverse and gaming integration.

---

## 🎯 Primary Project Objectives

1.  **Geometric Precision**: Achieve sub-millimeter accurate facial reconstruction using the Basel Face Model (BFM).
2.  **Aesthetic Continuity**: Leverage **indianface LoRA** to ensure 2D and 3D outputs reflect authentic Indian features and styles.
3.  **High-Fidelity Smoothing**: Implement non-linear mesh post-processing to eliminate "stair-stepping" and faceted artifacts.
4.  **Hardware Democratization**: Ensure the entire high-end pipeline (Diffusion + 3D) runs on **consumer-tier GPUs (4GB VRAM)**.
5.  **Multi-View Structural Integrity**: Synthesize 15+ accurate views including profiles and back-of-head textures without identity drift.

---

## 🏗️ Comprehensive ML Workflow & Architecture

The system operates as a **Directed Acyclic Graph (DAG)** of specialized neural and geometric nodes, ensuring that visual synthesis is strictly constrained by 3D geometry.

### 1. Global Pipeline Architecture
```mermaid
graph TD
    %% Input Layer
    INPUT[Input Photo] --> ALIGN[Landmark Alignment - MediaPipe]

    subgraph "Spatial Analysis Tier"
        ALIGN --> REG[3DDFA_V2 Parameter Regression]
        REG --> BFM_GEN[BFM Dense Mesh Assembly]
    end

    subgraph "Multi-View Logic (Perfection v2.0)"
        BFM_GEN --> CN_GUARDS[ControlNet-Canny Structural Hints]
        INPUT --> G_WARP[Geometric View Projections]
        CN_GUARDS --> SD_GEN[Stable Diffusion + indianface LoRA]
        G_WARP --> SD_GEN
        SD_GEN --> VIEWS["16x Identity-Preserved Views"]
    end

    subgraph "Geometric Refinement Tier"
        BFM_GEN --> SUBDIV[Loop-Style Subdivision]
        SUBDIV --> LAPLAC[Laplacian Smoothing Filter]
        LAPLAC --> SMOOTH_MESH[Premium Curved Surface]
    end

    %% Fusion Layer
    VIEWS --> TEX[Intelligent Texture Composer]
    SMOOTH_MESH --> TEX
    
    %% Final
    TEX --> FINAL[Final 3D Avatar Output]
    FINAL --> EXPORT[GLB / OBJ / PLY]

    style INPUT fill:#f9f,stroke:#333
    style FINAL fill:#00ff00,stroke:#333,stroke-width:3px
```

### 2. View Synthesis Control Sequence
We use a **Double-Guarded Diffusion** strategy to generate profiles (Yaw 45-90°) and back-of-head views.

```mermaid
sequenceDiagram
    participant P as User Image
    participant W as Geometric Warper
    participant C as ControlNet (Structural Guard)
    participant D as Diffusion Engine (PEFT)
    participant R as Restoration Post-Process

    P->>W: Project source image to target Euler angles
    W->>C: Extract Canny edges from geometric hint
    C->>D: Inject structural constraints (Stencil)
    D->>R: Raw Synthesized Texture (20 steps)
    R->>R: Apply Unsharp Mask & CLAHE (Contrast Fix)
    Note over R: Result: High-fidelity texture source
```

---

## 🔬 Core Algorithms & Innovations

### 📐 3DMM Geometric Regression (The "Skeleton")
The system regresses a **62-dimensional parameter vector** representing Pose, Shape, and Expression.
*   **Pose ($\phi$)**: Maps the 3D model into the 2D image space (12-dim).
*   **Shape ($\alpha$)**: Coefficients for the principal components of facial structure (40-dim).
*   **Expression ($\beta$)**: Action units for eyes, mouth, and brows (10-dim).

**The Equation**: $\mathbf{S} = \mathbf{\bar{S}} + \mathbf{A}_{shape}\alpha + \mathbf{A}_{exp}\beta$
This allows the user to live-tune the mesh using the Gradio sliders while maintaining a mathematically sound facial structure.

### 🌀 Perfection v2.0: Mesh Refiner
To achieve professional-grade results, we implemented:
*   **Subdivision Surface**: Increases vertex density by interpolating interstitial points between low-poly triangles.
*   **Laplacian Smoothing**: A differential surface filter that solves for vertex positions where $\Delta x = 0$, removing "pointy" artifacts without sacrificing the unique jawline of the user.

### 🎨 Intelligent Texture Blending
The **TextureComposer** uses sophisticated sampling to blend 16 views:
*   **Luminance Matching**: Histogram normalization ensures that synthetic side views matches the original photo's lighting.
*   **Sharp Blending**: $Weight = Confidence^{3.0}$ — using a high confidence power ensures that only the best camera view colors each region, preventing "ghosting" artifacts.

---

## 🛠️ Technical Stack & Library Rationales

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Geometry** | 3DDFA_V2 | Industry standard for fastest, most accurate real-time facial alignment. |
| **Core Model** | BFM (Basel Face Model) | High vertex density (38K+) and standard UV coordinates for seamless texturing. |
| **Synthesis** | ControlNet-Canny | Necessary for **structural integrity**; forces AI to follow the exact face silhouette. |
| **Diffusion** | Stable Diffusion XL | Handles high-resolution texture synthesis and complex LoRA style injection. |
| **PEFT** | LoRA (indianface) | Enables the "Indian Identity" pass without full model retraining. |
| **Geometry Processing**| Trimesh | Powerful library for mesh refinement, smoothing, and OBJ/GLB exports. |
| **Computer Vision** | MediaPipe | Used for rapid ROI detection and landmark-based alignment pre-processing. |
| **Interaction** | Gradio 5.x | State-of-the-art UI framework for real-time slider manipulation and 3D previewing. |

---

## 💾 Hardware Optimizations (Windows / RTX 30-series)
The project is uniquely optimized for **4GB VRAM** users:
*   **Sequential CPU Offload**: Models are only moved to GPU memory right before the forward pass.
*   **VAE Tiling**: Processes high-res image decoding in small chunks to prevent VRAM crashes.
*   **ONNX Serve**: Geometry regression happens via ONNX Runtime for 10x speedup on CPUs.

---

## 🚀 Presentation Walkthrough

### 1. Preparation
1.  Initialize environment: `python scripts/setup_venv.py`
2.  Fix conflicts: `python scripts/fix_dependencies.py`
3.  Launch: `python app.py`

### 2. Live Demo Steps
1.  **The Upload**: Show how MediaPipe detects the face and centers it.
2.  **2D Transformation**: Demonstrate the Indian Face LoRA changing a photo into a stylized avatar.
3.  **3D Generation**: Trigger the **Perfection v2.0** pipeline and show the 16-view slider.
4.  **Tuning**: Move the "Expression" sliders to see the BFM mesh respond in real-time.
5.  **Inspection**: Inspect the UV texture map to see the seamless blending achieved by **TextureComposer**.

---

## 📊 Project Status & Future Work
- [x] **Phase 1**: Base 3D Reconstruction (Complete)
- [x] **Phase 2**: Multi-View Synthesis Engine (Complete)
- [x] **Phase 3**: Perfection v2.0 Refinement (Complete)
- [ ] **Phase 4**: Real-time Facial Animation & Rigging (Planned)
- [ ] **Phase 5**: Ethnic-specific Clothing (Sari/Sherwani) LoRA Extensions (Planned)

---
*Developed by the **GUNI Research Intern Team***
*Technical Infrastructure by **Antigravity AI***
