"""
MultiViewGenerator — Synthesizes 15+ consistent facial views from a single photo.

Uses Stable Diffusion img2img with angle-specific prompts for realistic view
synthesis. Falls back to geometric affine warps on CPU or when models are
not available.

Camera angles generated:
  Front (0°), Left/Right 30°, 45°, 60°, 90°,
  Top-Left, Top-Right, Top, Bottom,
  Back-Left, Back-Right, Back
"""

import os
import time
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ViewSpec:
    """Specification for a single synthesized view."""
    name: str
    yaw: float        # horizontal rotation in degrees (-180 to 180)
    pitch: float      # vertical rotation in degrees (-90 to 90)
    prompt_suffix: str # angle-specific text for diffusion prompt
    strength: float   # denoising strength (lower = more faithful to input)


# ── 15+ View Definitions ──
VIEW_SPECS = [
    ViewSpec("front",         0,    0,  "front facing portrait, looking directly at camera", 0.30),
    ViewSpec("left_30",     -30,    0,  "slightly turned left, 30 degree angle side view", 0.45),
    ViewSpec("right_30",     30,    0,  "slightly turned right, 30 degree angle side view", 0.45),
    ViewSpec("left_45",     -45,    0,  "turned left, 45 degree angle semi-profile view", 0.55),
    ViewSpec("right_45",     45,    0,  "turned right, 45 degree angle semi-profile view", 0.55),
    ViewSpec("left_60",     -60,    0,  "left side profile, 60 degree angle view", 0.60),
    ViewSpec("right_60",     60,    0,  "right side profile, 60 degree angle view", 0.60),
    ViewSpec("left_90",     -90,    0,  "full left side profile view, 90 degrees, ear visible", 0.65),
    ViewSpec("right_90",     90,    0,  "full right side profile view, 90 degrees, ear visible", 0.65),
    ViewSpec("top_left",    -45,   30,  "top left angle looking down, 45 degree yaw, elevated view", 0.60),
    ViewSpec("top_right",    45,   30,  "top right angle looking down, 45 degree yaw, elevated view", 0.60),
    ViewSpec("top",           0,   60,  "top of head view, bird's eye, looking down at crown", 0.70),
    ViewSpec("bottom",        0,  -45,  "bottom angle looking up, chin and jaw visible", 0.65),
    ViewSpec("back_left",  -135,    0,  "back left view, showing back of head and left ear", 0.75),
    ViewSpec("back_right",  135,    0,  "back right view, showing back of head and right ear", 0.75),
    ViewSpec("back",        180,    0,  "complete back of head view, showing hair and nape of neck", 0.80),
]


class MultiViewGenerator:
    """
    Generates 15+ consistent multi-angle facial views from a single input photo.

    Strategy:
      - Near-frontal views (±30°, ±45°): Use low denoising strength to preserve
        identity strongly from the input image.
      - Side views (±60°, ±90°): Use moderate strength with detailed angle prompts.
      - Extreme views (back, top): Use higher strength since these regions have
        no direct pixel correspondence in the input.
      - CPU fallback: Geometric affine warps for all views.
    """

    def __init__(self, device=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self._pipe_loaded = False
        print(f"MultiViewGenerator initialized on {self.device}")
        print(f"  -> {len(VIEW_SPECS)} view angles defined")

    # ──────────────────────────────────────
    #  Pipeline Loading
    # ──────────────────────────────────────
    def _load_pipeline(self):
        """Load Stable Diffusion img2img pipeline for view synthesis."""
        if self._pipe_loaded:
            return self.pipe is not None

        self._pipe_loaded = True  # prevent repeated attempts

        if self.device == "cpu":
            print("  [!] CPU mode - using geometric fallback (no diffusion)")
            return False

        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

            model_id = "Lykon/AnyLoRA"
            print(f"  Loading SD pipeline ({model_id}) for view synthesis...")
            t0 = time.time()

            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            # VRAM optimization for 4GB GPUs
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing("max")
            self.pipe.enable_vae_tiling()

            print(f"  [OK] Pipeline loaded in {time.time() - t0:.1f}s")
            return True

        except Exception as e:
            print(f"  [x] Pipeline load failed: {e}")
            print("  -> Falling back to geometric transforms")
            self.pipe = None
            return False

    # ──────────────────────────────────────
    #  Diffusion-Based View Synthesis
    # ──────────────────────────────────────
    def _generate_view_diffusion(self, face_image: Image.Image, spec: ViewSpec) -> Image.Image:
        """Generate a single view using Stable Diffusion img2img."""
        import torch

        base_prompt = (
            "same person, same face, same identity, consistent appearance, "
            "photorealistic portrait, natural skin texture, "
        )
        prompt = base_prompt + spec.prompt_suffix
        prompt += ", (identity preserved:1.4), high quality, detailed, 8k"

        negative_prompt = (
            "different person, different face, deformed, ugly, blurry, "
            "low quality, watermark, text, logo, cartoon, anime, "
            "extra limbs, mutation, disfigured"
        )

        # Prepare input — apply geometric pre-rotation to guide the diffusion
        input_img = self._apply_geometric_hint(face_image, spec)
        input_img = input_img.convert("RGB").resize((512, 512), Image.LANCZOS)

        try:
            torch.cuda.empty_cache()
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    image=input_img,
                    strength=spec.strength,
                    guidance_scale=7.0,
                    negative_prompt=negative_prompt,
                    num_inference_steps=15,
                ).images[0]
            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [!] OOM for {spec.name}, using geometric fallback")
                torch.cuda.empty_cache()
                return self._generate_view_geometric(face_image, spec)
            raise

    # ──────────────────────────────────────
    #  Geometric Fallback View Synthesis
    # ──────────────────────────────────────
    def _generate_view_geometric(self, face_image: Image.Image, spec: ViewSpec) -> Image.Image:
        """
        Generate an approximate view using affine/perspective transforms.
        Used as fallback when diffusion pipeline is unavailable.
        """
        img = np.array(face_image.convert("RGB").resize((512, 512), Image.LANCZOS))
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        yaw_rad = np.radians(spec.yaw)
        pitch_rad = np.radians(spec.pitch)

        # Perspective warp simulation
        # Horizontal rotation (yaw): compress one side, expand the other
        if abs(spec.yaw) <= 90:
            # Compute perspective transform points
            squeeze = np.cos(yaw_rad)  # side compression factor
            shift_x = np.sin(yaw_rad) * w * 0.15

            src_pts = np.float32([
                [0, 0], [w, 0], [w, h], [0, h]
            ])

            if spec.yaw > 0:  # turning right
                dst_pts = np.float32([
                    [shift_x, int(h * (1 - squeeze) * 0.3)],
                    [w, 0],
                    [w, h],
                    [shift_x, int(h - h * (1 - squeeze) * 0.3)]
                ])
            else:  # turning left
                dst_pts = np.float32([
                    [0, 0],
                    [w + shift_x, int(h * (1 - squeeze) * 0.3)],
                    [w + shift_x, int(h - h * (1 - squeeze) * 0.3)],
                    [0, h]
                ])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        elif abs(spec.yaw) > 90:
            # Back views: mirror + heavy blur + darken (simulate hair/back of head)
            result = cv2.flip(img, 1)
            result = cv2.GaussianBlur(result, (31, 31), 10)

            # Darken to simulate hair/back texture
            darken = 0.3 + 0.4 * (1 - abs(abs(spec.yaw) - 180) / 90)
            result = (result.astype(np.float32) * darken).clip(0, 255).astype(np.uint8)

            # Add brownish tint for hair
            hair_tint = np.array([40, 30, 20], dtype=np.float32)
            result = np.clip(result.astype(np.float32) + hair_tint, 0, 255).astype(np.uint8)
        else:
            result = img.copy()

        # Vertical rotation (pitch)
        if abs(spec.pitch) > 10:
            pitch_squeeze = np.cos(pitch_rad)
            if spec.pitch > 0:  # looking from above
                # Compress bottom, expand top
                src_p = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                offset = int(w * (1 - pitch_squeeze) * 0.2)
                dst_p = np.float32([
                    [0, 0], [w, 0],
                    [w - offset, h], [offset, h]
                ])
            else:  # looking from below
                src_p = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                offset = int(w * (1 - pitch_squeeze) * 0.2)
                dst_p = np.float32([
                    [offset, 0], [w - offset, 0],
                    [w, h], [0, h]
                ])
            M = cv2.getPerspectiveTransform(src_p, dst_p)
            result = cv2.warpPerspective(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Simulate lighting changes based on angle
        result = self._apply_angle_lighting(result, spec)

        return Image.fromarray(result)

    def _apply_geometric_hint(self, face_image: Image.Image, spec: ViewSpec) -> Image.Image:
        """Apply a mild geometric warp as a 'hint' for the diffusion model."""
        # For near-frontal views, just return the original
        if abs(spec.yaw) < 20 and abs(spec.pitch) < 15:
            return face_image

        # Apply a gentle version of the geometric transform
        img = np.array(face_image.convert("RGB").resize((512, 512), Image.LANCZOS))
        h, w = img.shape[:2]

        # Mild perspective shift — just enough to hint at the angle
        yaw_mild = spec.yaw * 0.3  # reduce intensity to 30%
        yaw_rad = np.radians(yaw_mild)
        squeeze = np.cos(yaw_rad)
        shift_x = np.sin(yaw_rad) * w * 0.08

        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        if yaw_mild > 0:
            dst_pts = np.float32([
                [shift_x, int(h * (1 - squeeze) * 0.15)],
                [w, 0], [w, h],
                [shift_x, int(h - h * (1 - squeeze) * 0.15)]
            ])
        else:
            dst_pts = np.float32([
                [0, 0],
                [w + shift_x, int(h * (1 - squeeze) * 0.15)],
                [w + shift_x, int(h - h * (1 - squeeze) * 0.15)],
                [0, h]
            ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        return Image.fromarray(result)

    def _apply_angle_lighting(self, img: np.ndarray, spec: ViewSpec) -> np.ndarray:
        """Simulate directional lighting based on view angle."""
        h, w = img.shape[:2]
        result = img.astype(np.float32)

        # Create a gradient mask to simulate side lighting
        if abs(spec.yaw) > 15:
            x = np.linspace(0, 1, w)
            if spec.yaw > 0:
                # Right turn - shadow on left side
                light_mask = 0.7 + 0.3 * x
            else:
                # Left turn - shadow on right side
                light_mask = 1.0 - 0.3 * x

            light_mask = light_mask[np.newaxis, :, np.newaxis]
            intensity = min(abs(spec.yaw) / 90.0, 1.0) * 0.5
            blended = result * (1.0 - intensity + intensity * light_mask)
            result = np.clip(blended, 0, 255)

        # Top/bottom lighting
        if abs(spec.pitch) > 15:
            y = np.linspace(0, 1, h)
            if spec.pitch > 0:
                light_mask = 0.8 + 0.2 * (1 - y)
            else:
                light_mask = 0.8 + 0.2 * y
            light_mask = light_mask[:, np.newaxis, np.newaxis]
            intensity = min(abs(spec.pitch) / 60.0, 1.0) * 0.3
            blended = result * (1.0 - intensity + intensity * light_mask)
            result = np.clip(blended, 0, 255)

        return result.astype(np.uint8)

    # ──────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────
    def generate_all_views(
        self,
        face_image: Image.Image,
        progress_callback=None,
    ) -> List[Tuple[Image.Image, ViewSpec]]:
        """
        Generate 15+ multi-angle views from a single face photo.

        Args:
            face_image: PIL image of aligned face (front-facing)
            progress_callback: Optional callable(current, total, name) for progress

        Returns:
            List of (PIL.Image, ViewSpec) tuples — one per camera angle
        """
        t0 = time.time()
        use_diffusion = self._load_pipeline()

        views = []
        total = len(VIEW_SPECS)

        for i, spec in enumerate(VIEW_SPECS):
            step_t0 = time.time()

            if progress_callback:
                progress_callback(i, total, spec.name)

            if use_diffusion and self.pipe is not None:
                print(f"  [{i+1:2d}/{total}] Synthesizing {spec.name:12s} "
                      f"(yaw={spec.yaw:+4.0f}, pitch={spec.pitch:+3.0f}, "
                      f"strength={spec.strength:.2f}) ...", end="")
                view_img = self._generate_view_diffusion(face_image, spec)
            else:
                print(f"  [{i+1:2d}/{total}] Warping {spec.name:12s} "
                      f"(yaw={spec.yaw:+4.0f}, pitch={spec.pitch:+3.0f}) ...", end="")
                view_img = self._generate_view_geometric(face_image, spec)

            print(f" {time.time() - step_t0:.1f}s")
            views.append((view_img, spec))

        elapsed = time.time() - t0
        method = "diffusion" if use_diffusion else "geometric"
        print(f"  [OK] {total} views generated in {elapsed:.1f}s ({method})")

        return views

    def generate_images_only(self, face_image: Image.Image) -> List[Image.Image]:
        """Convenience: return just the PIL images (no specs)."""
        views = self.generate_all_views(face_image)
        return [img for img, _ in views]

    def get_view_specs(self) -> List[ViewSpec]:
        """Return the list of view specifications."""
        return VIEW_SPECS.copy()


if __name__ == "__main__":
    gen = MultiViewGenerator()
    # Quick test with a dummy image
    dummy = Image.new("RGB", (512, 512), (180, 140, 120))
    views = gen.generate_all_views(dummy)
    print(f"Generated {len(views)} views")
    for img, spec in views:
        print(f"  {spec.name}: {img.size}")
