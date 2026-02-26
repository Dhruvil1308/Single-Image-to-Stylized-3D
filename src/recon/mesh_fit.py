"""
MeshFitter — now powered by 3DDFA_V2 BFM morphable model.

Produces dense, smooth, UV-textured 3D face meshes (38K vertices, 76K faces)
from a single photograph. Replaces the old Mediapipe landmark approach.
"""

import time
import numpy as np
import trimesh
from PIL import Image

from src.models.face_recon import FaceReconstructor


class MeshFitter:
    """
    3D face mesh fitting using BFM morphable model.
    Delegates all heavy lifting to FaceReconstructor.
    """

    def __init__(self):
        self.reconstructor = FaceReconstructor()
        print("MeshFitter ready (BFM morphable model).")

    def fit(self, views, flame_wrapper, num_iters=0):
        """
        Reconstruct a 3D face mesh from the input views.

        Args:
            views: list of PIL images (uses the first non-None image)
            flame_wrapper: (unused, kept for API compatibility)

        Returns:
            trimesh.Trimesh with UV texture
        """
        t0 = time.time()

        source = None
        for img in views:
            if img is not None:
                source = img
                break

        if source is None:
            print("No input image — returning default mesh.")
            return self._default_mesh()

        print("Reconstructing 3D face with BFM morphable model...")
        mesh = self.reconstructor.reconstruct(source)

        dt = time.time() - t0
        print(f"MeshFitter complete in {dt:.1f}s")
        return mesh

    def modify_params(self, shape_deltas=None, expr_deltas=None):
        """
        Modify the last reconstructed face and return updated mesh.
        Called by customization sliders.
        """
        return self.reconstructor.modify_params(
            shape_deltas=shape_deltas,
            expr_deltas=expr_deltas
        )

    def _default_mesh(self):
        return trimesh.creation.icosphere(subdivisions=3, radius=60)


if __name__ == "__main__":
    fitter = MeshFitter()
    dummy = Image.new("RGB", (512, 512), (128, 100, 80))
    mesh = fitter.fit([dummy], None)
    print(f"Test: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
