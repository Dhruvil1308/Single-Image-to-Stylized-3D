import gradio as gr
from PIL import Image
from src.preprocess.extractor import FaceExtractor
from src.preprocess.segment import Segmenter
from src.models.anime_gen import AnimeGenerator
from src.models.flame_wrapper import FLAMEWrapper
from src.models.morphable_diffusion import MorphableDiffusion
from src.recon.mesh_fit import MeshFitter
from src.recon.exporter import Exporter
import os

# Initialize modules
extractor = FaceExtractor()
segmenter = Segmenter()
anime_gen = AnimeGenerator()
flame = FLAMEWrapper()
mv_synthesis = MorphableDiffusion()
fitter = MeshFitter()
exporter = Exporter()

def process_avatar(input_img, mode, style):
    if input_img is None:
        return None, "Please upload a photo."
    
    # save temp
    input_img.save("temp_input.jpg")
    
    # 1. Preprocess
    print("Starting preprocessing...")
    aligned_img = extractor.align_and_crop("temp_input.jpg")
    if aligned_img is None:
        return None, "No face detected. Please use a clear front-facing photo."
    
    # 2. Routing
    if mode == "2D Anime":
        print(f"Executing 2D Anime path with style: {style}...")
        result = anime_gen.generate(aligned_img, style=style)
        return result, f"2D Avatar ({style}) Generated Successfully!"
    
    else:
        print("Executing 3D Avatar path...")
        # 3. 3D Logic
        # 3a. Multi-view Synthesis
        views = mv_synthesis.generate_views(aligned_img)
        
        # 3b. Mesh Fitting
        mesh = fitter.fit(views, flame)
        
        # 3c. Export
        paths = exporter.export_3d(mesh)
        
        return views[0], f"3D Avatar Processed! \nViews synthesized: {len(views)} \nMesh fitted and exported to assets/ (Randomized for prototype)."

# UI Components
with gr.Blocks(title="Indian Avatar AI") as demo:
    gr.Markdown("# 🇮🇳 Indian Avatar AI: 2D/3D Generation")
    gr.Markdown("Convert a single photo into a stylized 2D Anime avatar or a 3D head model.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Front-facing Photo")
            style_opt = gr.Radio(["Realistic", "Cartoon", "Stylized"], label="Style", value="Stylized")
            mode_opt = gr.Radio(["2D Anime", "3D Avatar"], label="Output Type", value="2D Anime")
            generate_btn = gr.Button("Generate Avatar", variant="primary")
            
        with gr.Column():
            output_display = gr.Image(label="Avatar Preview")
            status_text = gr.Textbox(label="Status", interactive=False)
            
    with gr.Accordion("3D Customization & Emotions", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Emotions")
                with gr.Row():
                    happy_btn = gr.Button("😊 Happy")
                    sad_btn = gr.Button("😢 Sad")
                    angry_btn = gr.Button("😠 Angry")
            with gr.Column():
                gr.Markdown("### Customization")
                nose_slider = gr.Slider(-1, 1, 0, label="Nose Size")
                jaw_slider = gr.Slider(-1, 1, 0, label="Jaw Width")
                eye_slider = gr.Slider(-1, 1, 0, label="Eye Shape")
                
    generate_btn.click(
        process_avatar,
        inputs=[input_image, mode_opt, style_opt],
        outputs=[output_display, status_text]
    )

if __name__ == "__main__":
    # max_file_size=10mb to prevent protocol errors with large images
    demo.launch(max_file_size="10mb")
