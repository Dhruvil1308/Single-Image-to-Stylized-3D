"""
MeshFitter — 3D face mesh fitting using BFM morphable model.

Now supports multi-view texture composition: when multiple synthesized
views are available (from MorphableDiffusion), the fitter passes them
to the TextureComposer for high-quality UV texture generation.
"""

import time
import numpy as np
import trimesh
from PIL import Image
from typing import List, Optional, Tuple

from src.models.face_recon import FaceReconstructor


class MeshFitter:
    """
    3D face mesh fitting using BFM morphable model.
    Delegates geometry reconstruction to FaceReconstructor and
    optionally enhances texture using multi-view composition.
    """

    def __init__(self):
        self.reconstructor = FaceReconstructor()
        self._texture_composer = None
        print("MeshFitter ready (BFM morphable model + multi-view texture).")

    def _get_texture_composer(self):
        """Lazy-load the TextureComposer."""
        if self._texture_composer is None:
            from src.recon.texture_composer import TextureComposer
            self._texture_composer = TextureComposer(tex_size=1024)
        return self._texture_composer

    def fit(
        self,
        views: List[Image.Image],
        flame_wrapper,
        num_iters: int = 0,
        views_with_angles: Optional[List[Tuple[Image.Image, float, float]]] = None,
    ) -> trimesh.Trimesh:
        """
        Reconstruct a 3D face mesh from the input views.

        Args:
            views: list of PIL images (uses the first non-None image for geometry)
            flame_wrapper: (unused, kept for API compatibility)
            num_iters: (unused, kept for API compatibility)
            views_with_angles: Optional list of (PIL.Image, yaw, pitch) from
                              MorphableDiffusion for multi-view texture composition

        Returns:
            trimesh.Trimesh with UV texture (multi-view if available)
        """
        t0 = time.time()

        # Find the primary (front-facing) image for geometry reconstruction
        source = None
        for img in views:
            if img is not None:
                source = img
                break

        if source is None:
            print("No input image — returning default mesh.")
            return self._default_mesh()

        # Step 1: Reconstruct geometry from the front view
        print("Reconstructing 3D face geometry with BFM morphable model...")
        mesh = self.reconstructor.reconstruct(source)

        # Step 2: If multi-view data is available, enhance the texture
        if views_with_angles and len(views_with_angles) > 1:
            print(f"Enhancing texture with {len(views_with_angles)}-view composition...")
            mesh = self._apply_multiview_texture(mesh, views_with_angles)

        dt = time.time() - t0
        print(f"MeshFitter complete in {dt:.1f}s")
        return mesh

    def _apply_multiview_texture(
        self,
        mesh: trimesh.Trimesh,
        views_with_angles: List[Tuple[Image.Image, float, float]],
    ) -> trimesh.Trimesh:
        """
        Re-texture the mesh using multi-view composition.
        Replaces the single-view texture with a blended multi-view texture.
        """
        try:
            composer = self._get_texture_composer()
            recon = self.reconstructor

            # We need: vertices in image space, UV coords, faces
            if (recon.current_param is None or
                recon.uv_coords is None or
                recon.current_img is None):
                print("  [!] Reconstruction data missing, keeping single-view texture")
                return mesh

            # Reconstruct vertices in image space for texture sampling
            vertices = recon._reconstruct_vertices(
                recon.current_param, recon.current_roi_box
            )

            h, w = recon.current_img.shape[:2]

            # Compose multi-view texture
            composed_tex = composer.compose_texture(
                views_with_angles=views_with_angles,
                vertices_3xN=vertices,
                uv_coords=recon.uv_coords,
                faces=recon.tri,
                img_shape=(h, w),
            )

            if composed_tex is not None:
                # Replace the mesh texture with the composed one
                import trimesh.visual
                import trimesh.visual.material

                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=composed_tex,
                    metallicFactor=0.0,
                    roughnessFactor=0.7,
                )
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=recon.uv_coords[:len(mesh.vertices)],
                    material=material,
                )
                print("  [OK] Multi-view texture applied to mesh")
            else:
                print("  [!] Texture composition returned None, keeping original")

        except Exception as e:
            print(f"  [!] Multi-view texture failed: {e}")
            print("  -> Keeping single-view texture")

        return mesh

    def modify_params(self, shape_deltas=None, expr_deltas=None):
        """
        Modify the last reconstructed face and return updated mesh.
        Called by customization sliders.
        """
        return self.reconstructor.modify_params(
            shape_deltas=shape_deltas,
            expr_deltas=expr_deltas,
        )

    def _default_mesh(self):
        return trimesh.creation.icosphere(subdivisions=3, radius=60)


if __name__ == "__main__":
    fitter = MeshFitter()
    dummy = Image.new("RGB", (512, 512), (128, 100, 80))
    mesh = fitter.fit([dummy], None)
    print(f"Test: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
