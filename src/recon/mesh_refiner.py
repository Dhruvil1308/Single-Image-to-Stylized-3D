"""
MeshRefiner — Post-processing for 3D face geometry.

Provides Laplacian smoothing to reduce reconstruction noise and mesh 
subdivision (Loop-style) to increase vertex density for a premium, 
high-resolution appearance.
"""

import numpy as np
import trimesh
import cv2

class MeshRefiner:
    """
    Handles smoothing and subdivision of 3D face meshes.
    """

    def __init__(self, subdivision_levels: int = 1, smooth_iterations: int = 10):
        self.subdivision_levels = subdivision_levels
        self.smooth_iterations = smooth_iterations
        print(f"MeshRefiner initialized (Subdiv: {subdivision_levels}, Smooth: {smooth_iterations})")

    def refine(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Apply smoothing and subdivision to the input mesh.
        """
        print(f"  Refining mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces...")

        # 1. Laplacian Smoothing
        # Reduces the 'stair-stepping' artifacts from 3DDFA regression
        if self.smooth_iterations > 0:
            mesh = trimesh.smoothing.filter_laplacian(
                mesh, 
                iterations=self.smooth_iterations,
                lamb=0.5
            )

        # 2. Subdivision
        # Increases vertex density for a smoother silhoutte
        if self.subdivision_levels > 0:
            for _ in range(self.subdivision_levels):
                # Using simple linear subdivision + smoothing to simulate Loop
                # as trimesh.subdivision.subdivide is purely linear
                mesh = mesh.subdivide()
                # Apply a light smoothing after subdivide to round out the new faces
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=2)

        print(f"  [OK] Refinement complete: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return mesh

    def fix_normals(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Ensure normals are consistent and pointing outwards."""
        mesh.fix_normals()
        return mesh
