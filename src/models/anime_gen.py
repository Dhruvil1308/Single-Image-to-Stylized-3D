import torch
from PIL import Image
import numpy as np
import os
import time

class AnimeGenerator:
    def __init__(self, model_id="Lykon/AnyLoRA", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_id = model_id
        self.pipe = None
        print(f"Initializing AnimeGenerator with {model_id} on {device}")
        
    def load_pipeline(self):
        """Loads the SD pipeline optimized for 4GB VRAM with fixed LoRA loading."""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
            
            print(f"[1/4] Loading model: {self.model_id}...")
            t0 = time.time()
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            print(f"[2/4] Model loaded in {time.time()-t0:.0f}s")
            
            # Fast scheduler — DPM++ gives good quality in fewer steps
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Load LoRA using PEFT (must be done BEFORE offloading)
            lora_path = "models/indianface_lora"
            if os.path.exists(lora_path):
                print(f"[3/4] Loading Indian Face LoRA...")
                try:
                    from peft import PeftModel
                    self.pipe.unet = PeftModel.from_pretrained(
                        self.pipe.unet, lora_path
                    )
                    self.pipe.unet.eval()
                    print("       ✅ LoRA loaded successfully!")
                except Exception as e:
                    print(f"       ⚠️ LoRA skipped: {e}")
            else:
                print("[3/4] No LoRA found, using base model")

            # VRAM strategy for 4GB GPU:
            # model_cpu_offload is MUCH faster than sequential_cpu_offload
            # It moves entire model components (not individual layers)
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing("max")
                self.pipe.enable_vae_tiling()
                print("[4/4] ✅ VRAM optimized (model offload + attn slicing + VAE tiling)")
            else:
                self.pipe = self.pipe.to(self.device)
                print("[4/4] Running on CPU")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline FAILED: {e}")
            import traceback
            traceback.print_exc()
            self.pipe = None
            return False

    def generate(self, face_image, style="Stylized", negative_prompt=None):
        
        if self.pipe is None:
            print("=" * 40)
            print("Loading pipeline (first time only)...")
            print("=" * 40)
            if not self.load_pipeline():
                return face_image

        # Indian-optimized prompts — distinct looks for each style
        style_prompts = {
            "Stylized": (
                "stylized digital art portrait, indianface, beautiful indian features, "
                "vibrant colors, artistic illustration, hand-painted look, "
                "dramatic lighting, detailed brushstrokes, concept art style"
            ),
            "Cartoon": (
                "pixar 3d cartoon character, indianface, cute indian cartoon face, "
                "big expressive eyes, smooth 3d render, colorful, fun character design, "
                "disney pixar style, rounded features, animated movie character"
            ),
        }

        prompt = style_prompts.get(style, style_prompts["Stylized"])
        prompt += ", (indian heritage:1.2), detailed eyes, masterpiece, best quality"

        # Style-specific negative prompts
        style_negatives = {
            "Stylized": "photo, photograph, realistic, 3d render, low quality, blurry, deformed",
            "Cartoon": "realistic, photo, photograph, painting, sketch, low quality, blurry, deformed",
        }
        if negative_prompt is None:
            negative_prompt = style_negatives.get(style, style_negatives["Stylized"])

        # Cartoon uses higher strength for more transformation
        strength = 0.75 if style == "Cartoon" else 0.65

        input_image = face_image.convert("RGB").resize((512, 512))

        print(f"Generating {style} avatar...")
        t0 = time.time()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=7.5,
                    negative_prompt=negative_prompt,
                    num_inference_steps=12
                ).images[0]
            
            elapsed = time.time() - t0
            print(f"✅ Done in {elapsed:.1f}s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ GPU OOM — retrying with smaller image...")
                torch.cuda.empty_cache()
                input_small = face_image.convert("RGB").resize((384, 384))
                with torch.inference_mode():
                    result = self.pipe(
                        prompt=prompt,
                        image=input_small,
                        strength=0.6,
                        guidance_scale=6.0,
                        negative_prompt=negative_prompt,
                        num_inference_steps=8
                    ).images[0]
                result = result.resize((512, 512), Image.LANCZOS)
                print(f"✅ Done (fallback) in {time.time()-t0:.1f}s")
            else:
                print(f"❌ Error: {e}")
                return face_image
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result

if __name__ == "__main__":
    gen = AnimeGenerator()
    print("AnimeGenerator ready.")
