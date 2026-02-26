import gradio as gr
import torch
import numpy as np
import os
import time
import logging
import warnings

# Suppress noisy but harmless warnings
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message=".*CLIPFeatureExtractor.*")
from PIL import Image
from src.preprocess.extractor import FaceExtractor
from src.preprocess.segment import Segmenter
from src.models.anime_gen import AnimeGenerator
from src.models.flame_wrapper import FLAMEWrapper
from src.models.morphable_diffusion import MorphableDiffusion
from src.recon.mesh_fit import MeshFitter
from src.recon.exporter import Exporter

# ─────────────────────────────────────────────────────────────
#  Initialize Modules
# ─────────────────────────────────────────────────────────────
extractor = FaceExtractor()
segmenter = Segmenter()
anime_gen = AnimeGenerator()
flame = FLAMEWrapper()
mv_synthesis = MorphableDiffusion()
fitter = MeshFitter()
exporter = Exporter()


# ─────────────────────────────────────────────────────────────
#  Core Processing
# ─────────────────────────────────────────────────────────────

def preprocess_face(input_img):
    """Detect and align face from uploaded photo."""
    if input_img is None:
        return None, "Please upload a photo first."
    input_img.save("temp_input.jpg")
    aligned = extractor.align_and_crop("temp_input.jpg")
    if aligned is None:
        return None, "No face detected. Please use a clear front-facing photo."
    return aligned, "Face detected and aligned successfully."


# ─────────────────────────────────────────────────────────────
#  Navigation Handlers
# ─────────────────────────────────────────────────────────────

def on_image_upload(input_img):
    """Validate uploaded image and show mode selection."""
    if input_img is None:
        return (
            gr.update(visible=False),                    # mode_section
            gr.update(visible=False),                    # section_2d
            gr.update(visible=False),                    # section_3d
            gr.update(value="", visible=False),          # upload_status
        )

    aligned, msg = preprocess_face(input_img)
    if aligned is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=msg, visible=True),
        )

    return (
        gr.update(visible=True),                         # show mode selection
        gr.update(visible=False),                        # hide 2D section
        gr.update(visible=False),                        # hide 3D section
        gr.update(
            value="Face detected successfully! Choose a generation mode below.",
            visible=True
        ),
    )


def select_2d_mode():
    """Switch to 2D generation section."""
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def select_3d_mode():
    """Switch to 3D generation section."""
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def go_back_to_modes():
    """Return to mode selection."""
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


# ─────────────────────────────────────────────────────────────
#  2D Generation Handlers
# ─────────────────────────────────────────────────────────────

def generate_stylized(input_img):
    """Generate a stylized artistic image from the uploaded face."""
    if input_img is None:
        return None, "Please upload a photo first."

    t_start = time.time()
    aligned, msg = preprocess_face(input_img)
    if aligned is None:
        return None, msg

    print("Generating Stylized Image...")
    result = anime_gen.generate(aligned, style="Stylized")
    elapsed = time.time() - t_start

    os.makedirs("assets", exist_ok=True)
    result.save(os.path.join("assets", "stylized_output.png"))

    return result, f"Stylized image generated in {elapsed:.1f}s"


def generate_cartoon(input_img):
    """Generate a cartoon-style image from the uploaded face."""
    if input_img is None:
        return None, "Please upload a photo first."

    t_start = time.time()
    aligned, msg = preprocess_face(input_img)
    if aligned is None:
        return None, msg

    print("Generating Cartoon Image...")
    result = anime_gen.generate(aligned, style="Cartoon")
    elapsed = time.time() - t_start

    os.makedirs("assets", exist_ok=True)
    result.save(os.path.join("assets", "cartoon_output.png"))

    return result, f"Cartoon image generated in {elapsed:.1f}s"


# ─────────────────────────────────────────────────────────────
#  3D Generation Handlers
# ─────────────────────────────────────────────────────────────

def generate_3d(input_img):
    """Generate a 3D avatar model with multi-view synthesis and texture composition."""
    if input_img is None:
        return None, None, None, None, None, "Please upload a photo first."

    t_start = time.time()
    aligned, msg = preprocess_face(input_img)
    if aligned is None:
        return None, None, None, None, None, msg

    # Step 1: Generate 15+ multi-angle views
    print("="*60)
    print("STEP 1/3: Multi-View Synthesis (15+ angles)...")
    print("="*60)
    views = mv_synthesis.generate_views(aligned)
    n_views = len(views)
    t_views = time.time() - t_start

    # Get angle metadata for texture composition
    views_with_angles = mv_synthesis.get_cached_views_with_angles()

    # Step 2: Fit mesh with multi-view texture
    print("="*60)
    print("STEP 2/3: 3D Mesh Reconstruction + Multi-View Texture...")
    print("="*60)
    mesh = fitter.fit(views, flame, views_with_angles=views_with_angles)

    # Step 3: Export
    print("="*60)
    print("STEP 3/3: Exporting 3D Model...")
    print("="*60)
    exporter.export_3d(mesh)
    elapsed = time.time() - t_start

    glb_path = os.path.abspath(os.path.join("assets", "avatar.glb"))
    obj_path = os.path.abspath(os.path.join("assets", "avatar.obj"))

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    has_tex = getattr(mesh.visual, 'kind', 'none') == 'texture'
    tex_info = "Multi-View UV-Textured" if has_tex else "Vertex Colors"

    # Build gallery data: list of (image, label) for Gradio Gallery
    gallery_data = []
    if views_with_angles:
        for img, yaw, pitch in views_with_angles:
            label = f"Yaw {yaw:+.0f}° Pitch {pitch:+.0f}°"
            gallery_data.append((img, label))
    else:
        for i, v in enumerate(views):
            gallery_data.append((v, f"View {i+1}"))

    status = (
        f"✅ 3D Avatar generated in {elapsed:.1f}s\n"
        f"Views Synthesized: {n_views} | View Generation: {t_views:.1f}s\n"
        f"Vertices: {n_verts:,} | Faces: {n_faces:,} | {tex_info}\n"
        f"Formats: OBJ, GLB (Download below)"
    )

    return gallery_data, views[0], glb_path, glb_path, obj_path, status


def update_customization(nose, jaw, eye, emotion):
    """Update 3D mesh by modifying BFM shape/expression parameters."""
    print(f"Customizing: Nose={nose}, Jaw={jaw}, Eye={eye}, Emotion={emotion}")

    # Map sliders to BFM shape parameter indices
    shape_deltas = {}
    if nose != 0:
        shape_deltas[3] = nose       # nose bridge height
    if jaw != 0:
        shape_deltas[2] = jaw        # jaw width
    if eye != 0:
        shape_deltas[4] = eye        # forehead/eye depth

    # Map emotion to expression parameters
    expr_deltas = {}
    emotion_map = {
        "happy": {1: 2.0, 3: 1.0},    # smile + eye squint
        "sad": {1: -1.5, 0: -0.5},     # mouth down + brow lower
        "angry": {0: 2.0, 4: 1.0},     # brow lower + mouth tight
        "surprised": {0: 2.0, 2: 2.0}, # brow raise + mouth open
    }
    if emotion and emotion.lower() in emotion_map:
        expr_deltas = emotion_map[emotion.lower()]

    # Regenerate mesh with modified parameters
    mesh = fitter.modify_params(
        shape_deltas=shape_deltas if shape_deltas else None,
        expr_deltas=expr_deltas if expr_deltas else None
    )

    if mesh is None:
        return None, None, None, "Generate a 3D avatar first, then customize."

    exporter.export_3d(mesh)

    glb_path = os.path.abspath(os.path.join("assets", "avatar.glb"))
    obj_path = os.path.abspath(os.path.join("assets", "avatar.obj"))

    return glb_path, glb_path, obj_path, "✅ Customization applied (BFM parametric update)."



# ─────────────────────────────────────────────────────────────
#  UI / UX
# ─────────────────────────────────────────────────────────────

custom_css = """
/* ── Header ── */
.main-title {
    text-align: center;
    font-size: 2.2em !important;
    margin-bottom: 2px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 15px;
    margin-top: 0 !important;
    margin-bottom: 20px !important;
}

/* ── Upload area ── */
.upload-area { max-width: 480px; margin: 0 auto; }
.upload-status {
    text-align: center;
    font-size: 14px;
    padding: 10px 16px;
    border-radius: 8px;
    margin-top: 8px;
}

/* ── Mode selection cards ── */
.mode-card {
    border-radius: 16px !important;
    padding: 8px !important;
    text-align: center;
    transition: transform 0.2s ease;
}
.mode-card:hover { transform: translateY(-2px); }
.mode-btn {
    height: 52px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

/* ── Generation buttons ── */
.gen-btn {
    height: 50px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* ── Status ── */
.status-box { font-family: monospace; font-size: 13px; }

/* ── Back button ── */
.back-btn { max-width: 160px !important; }

/* ── Customization ── */
.customization-section {
    border-top: 1px solid #ddd;
    padding-top: 12px;
    margin-top: 16px;
}

/* ── Step labels ── */
.step-label {
    font-size: 1.1em !important;
    font-weight: 600;
    color: #444;
}

/* ── Section description ── */
.section-desc { color: #555; font-size: 14px; line-height: 1.6; }
"""


with gr.Blocks(
    title="Indian Avatar AI",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Header
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    gr.Markdown(
        "# Indian Avatar AI - Deep Accuracy 2.0",
        elem_classes="main-title"
    )
    gr.Markdown(
        "Transform your face photo into stunning 2D art or a realistic 3D avatar model.",
        elem_classes="subtitle"
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Step 1 : Upload Photo
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    gr.Markdown("### Step 1 : Upload Your Face Photo", elem_classes="step-label")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            input_image = gr.Image(
                type="pil",
                label="Front-facing Photo",
                height=320,
                sources=["upload", "webcam"],
                elem_classes="upload-area"
            )
            upload_status = gr.Markdown("", visible=False, elem_classes="upload-status")
        with gr.Column(scale=1):
            pass

    gr.Markdown("---")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Step 2 : Mode Selection  (hidden until upload)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with gr.Column(visible=False) as mode_section:
        gr.Markdown(
            "### Step 2 : Choose Generation Mode",
            elem_classes="step-label"
        )
        gr.Markdown(
            "Select what you want to create from your uploaded photo.",
            elem_classes="section-desc"
        )

        with gr.Row(equal_height=True):
            # ── Card 1 : 2D Image Generation ──
            with gr.Column(scale=1, elem_classes="mode-card"):
                with gr.Group():
                    gr.Markdown(
                        "#### 2D Image Generation\n\n"
                        "Transform your photo into **Stylized** digital art "
                        "or a fun **Cartoon** character."
                    )
                    btn_2d_mode = gr.Button(
                        "Select 2D Generation",
                        variant="primary",
                        elem_classes="mode-btn"
                    )

            # ── Card 2 : 3D Avatar Generation ──
            with gr.Column(scale=1, elem_classes="mode-card"):
                with gr.Group():
                    gr.Markdown(
                        "#### 3D Avatar Generation\n\n"
                        "Generate a full **3D model** (GLB / OBJ) that you can "
                        "rotate, view, and **download**."
                    )
                    btn_3d_mode = gr.Button(
                        "Select 3D Generation",
                        variant="primary",
                        elem_classes="mode-btn"
                    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2D Image Generation Section  (hidden until selected)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with gr.Column(visible=False) as section_2d:
        with gr.Row():
            back_btn_2d = gr.Button(
                "< Back", size="sm", variant="secondary",
                elem_classes="back-btn"
            )
            gr.Markdown("### 2D Image Generation")

        gr.Markdown(
            "Choose an art style below and click generate. "
            "You can switch between styles using the tabs.",
            elem_classes="section-desc"
        )

        with gr.Tabs():
            # ── Sub-option 1 : Stylized Image ──
            with gr.Tab("Stylized Image"):
                gr.Markdown(
                    "**Stylized Image** creates a vibrant digital-art illustration "
                    "with rich colours, artistic lighting, and a hand-painted feel."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        stylized_btn = gr.Button(
                            "Generate Stylized Image",
                            variant="primary",
                            elem_classes="gen-btn"
                        )
                        stylized_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_classes="status-box"
                        )
                    with gr.Column(scale=2):
                        stylized_output = gr.Image(
                            label="Stylized Output", height=420
                        )

            # ── Sub-option 2 : Cartoon Image ──
            with gr.Tab("Cartoon Image"):
                gr.Markdown(
                    "**Cartoon Image** turns your face into a fun Pixar / Disney-style "
                    "character with big expressive eyes and a smooth 3D-render look."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cartoon_btn = gr.Button(
                            "Generate Cartoon Image",
                            variant="primary",
                            elem_classes="gen-btn"
                        )
                        cartoon_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_classes="status-box"
                        )
                    with gr.Column(scale=2):
                        cartoon_output = gr.Image(
                            label="Cartoon Output", height=420
                        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  3D Avatar Generation Section  (hidden until selected)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with gr.Column(visible=False) as section_3d:
        with gr.Row():
            back_btn_3d = gr.Button(
                "< Back", size="sm", variant="secondary",
                elem_classes="back-btn"
            )
            gr.Markdown("### 3D Avatar Generation")

        gr.Markdown(
            "Generate your 3D avatar using **15+ multi-angle view synthesis**. "
            "The system creates left, right, semi-profile, top, and back-of-head views "
            "to produce a high-resolution 3D model with complete texture coverage.",
            elem_classes="section-desc"
        )

        with gr.Row():
            # ── Left column : Controls ──
            with gr.Column(scale=1):
                generate_3d_btn = gr.Button(
                    "Generate 3D Avatar (15+ Views)",
                    variant="primary",
                    elem_classes="gen-btn"
                )
                avatar_3d_status = gr.Textbox(
                    label="Pipeline Status",
                    interactive=False,
                    lines=4,
                    elem_classes="status-box"
                )

                # Download section
                gr.Markdown("#### Download 3D Model")
                download_glb = gr.File(
                    label="Download GLB",
                    interactive=False
                )
                download_obj = gr.File(
                    label="Download OBJ",
                    interactive=False
                )

                # Fine-tune customization
                gr.Markdown(
                    "#### Fine-Tune Your Avatar",
                    elem_classes="customization-section"
                )
                nose_slider = gr.Slider(
                    -2, 2, 0, step=0.1, label="Nose Bridge Height"
                )
                jaw_slider = gr.Slider(
                    -2, 2, 0, step=0.1, label="Jawline Width"
                )
                eye_slider = gr.Slider(
                    -2, 2, 0, step=0.1, label="Forehead / Eye Depth"
                )
                emotion_opt = gr.Dropdown(
                    ["none", "Happy", "Sad", "Angry", "Surprised"],
                    label="Expression",
                    value="none"
                )
                customize_btn = gr.Button(
                    "Apply Customization", variant="secondary"
                )

            # ── Right column : Preview & 3D Viewer ──
            with gr.Column(scale=2):
                avatar_preview = gr.Image(
                    label="Front Face Preview", height=220
                )
                model_viewer = gr.Model3D(
                    label="3D Model Viewer", height=400
                )

        # ── Multi-View Gallery (below the main row) ──
        gr.Markdown(
            "#### 🔄 Synthesized Multi-Angle Views",
            elem_classes="step-label"
        )
        gr.Markdown(
            "The views below were generated from your single photo — "
            "they are used to create the 3D texture map.",
            elem_classes="section-desc"
        )
        multiview_gallery = gr.Gallery(
            label="Multi-View Synthesis (15+ angles)",
            columns=5,
            rows=3,
            height=360,
            object_fit="contain",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Event Handlers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Upload → validate face → show mode selection
    input_image.change(
        fn=on_image_upload,
        inputs=[input_image],
        outputs=[mode_section, section_2d, section_3d, upload_status]
    )

    # Mode selection buttons
    btn_2d_mode.click(
        fn=select_2d_mode,
        outputs=[mode_section, section_2d, section_3d]
    )
    btn_3d_mode.click(
        fn=select_3d_mode,
        outputs=[mode_section, section_2d, section_3d]
    )

    # Back navigation
    back_btn_2d.click(
        fn=go_back_to_modes,
        outputs=[mode_section, section_2d, section_3d]
    )
    back_btn_3d.click(
        fn=go_back_to_modes,
        outputs=[mode_section, section_2d, section_3d]
    )

    # 2D generation
    stylized_btn.click(
        fn=generate_stylized,
        inputs=[input_image],
        outputs=[stylized_output, stylized_status]
    )
    cartoon_btn.click(
        fn=generate_cartoon,
        inputs=[input_image],
        outputs=[cartoon_output, cartoon_status]
    )

    # 3D generation (with multi-view gallery)
    generate_3d_btn.click(
        fn=generate_3d,
        inputs=[input_image],
        outputs=[
            multiview_gallery, avatar_preview,
            model_viewer,
            download_glb, download_obj,
            avatar_3d_status
        ]
    )

    # 3D customization
    customize_btn.click(
        fn=update_customization,
        inputs=[nose_slider, jaw_slider, eye_slider, emotion_opt],
        outputs=[model_viewer, download_glb, download_obj, avatar_3d_status]
    )


if __name__ == "__main__":
    print("Launching Indian Avatar AI Interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        max_file_size="10mb",
        allowed_paths=[os.path.abspath("assets")]
    )
