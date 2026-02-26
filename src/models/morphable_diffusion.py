"""
MorphableDiffusion — Multi-view synthesis orchestrator.

Generates 15+ consistent multi-angle facial images from a single input
photograph using GAN-based and diffusion-based neural networks. These
synthesized views are intelligently combined to produce a high-resolution
texture map used for 3D reconstruction.

Camera angles include:
  Left, Right, Semi-profile views (30°/45°/60°/90°),
  Top angles, Bottom, and Complete backside head textures.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

from src.synthesis.multi_view_generator import MultiViewGenerator, ViewSpec


class MorphableDiffusion:
    """
    Orchestrates multi-view facial synthesis for 3D texture reconstruction.

    This module serves as the high-level API between the Gradio UI and the
    underlying MultiViewGenerator engine. It manages:
      1. View generation (15+ angles) via diffusion or geometric fallback
      2. View metadata (angle labels) for the texture composer
      3. Progress reporting for the UI
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = MultiViewGenerator(device=self.device)
        self._last_views = None  # cache for texture composition
        print(f"MorphableDiffusion initialized on {self.device}.")
        print(f"  -> Ready for {len(self.generator.get_view_specs())}-view synthesis.")

    def generate_views(self, face_image: Image.Image) -> List[Image.Image]:
        """
        Generate multi-angle views from a single face photo.

        This is the main API called by app.py / generate_3d().
        Returns a list of PIL images — the first is always the front view.

        Also caches the full (image, spec) pairs for later use
        by the texture composer in mesh_fit.py.
        """
        print(f"MorphableDiffusion: Starting {len(self.generator.get_view_specs())}-view synthesis...")

        # Generate all views with angle metadata
        views_with_specs = self.generator.generate_all_views(face_image)
        self._last_views = views_with_specs

        # Return just the images (API-compatible with old code)
        images = [img for img, spec in views_with_specs]
        return images

    def generate_views_with_angles(
        self, face_image: Image.Image
    ) -> List[Tuple[Image.Image, float, float]]:
        """
        Generate views with angle metadata for the texture composer.

        Returns:
            List of (PIL.Image, yaw_degrees, pitch_degrees) tuples
        """
        views_with_specs = self.generator.generate_all_views(face_image)
        self._last_views = views_with_specs

        return [
            (img, spec.yaw, spec.pitch)
            for img, spec in views_with_specs
        ]

    def get_cached_views_with_angles(self) -> Optional[List[Tuple[Image.Image, float, float]]]:
        """
        Return the cached views from the last generate_views() call.
        Used by mesh_fit.py for texture composition without re-generating.
        """
        if self._last_views is None:
            return None

        return [
            (img, spec.yaw, spec.pitch)
            for img, spec in self._last_views
        ]

    def get_view_names(self) -> List[str]:
        """Return human-readable names for all view angles."""
        return [spec.name for spec in self.generator.get_view_specs()]

    def get_view_count(self) -> int:
        """Return the total number of views generated."""
        return len(self.generator.get_view_specs())


if __name__ == "__main__":
    md = MorphableDiffusion()
    print(f"MorphableDiffusion ready — {md.get_view_count()} views configured.")

    # Quick test with dummy image
    dummy = Image.new("RGB", (512, 512), (180, 140, 120))
    views = md.generate_views(dummy)
    print(f"Generated {len(views)} view images.")

    # Test with angles
    views_angles = md.get_cached_views_with_angles()
    if views_angles:
        for img, yaw, pitch in views_angles:
            print(f"  View: yaw={yaw:+.0f}° pitch={pitch:+.0f}° size={img.size}")
