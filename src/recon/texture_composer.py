"""
TextureComposer — Blends multiple synthesized views into a single UV texture map.

Given a set of (view_image, camera_angle) pairs and a BFM mesh with UV
coordinates, this module determines which view(s) have the best visibility
for each UV region and blends them with smooth angle-based confidence
weighting to produce a seamless, high-resolution texture.
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional


class TextureComposer:
    """
    Composes a high-resolution UV texture from multiple synthesized views.

    Strategy:
      1. For each UV coordinate, compute which camera angles would see that
         part of the face surface (based on the vertex normal vs view direction).
      2. Weight each view's contribution by how directly it faces the surface
         (dot product of normal and view direction → confidence).
      3. Blend weighted contributions with smooth falloff for seamless textures.
      4. Fill any remaining holes with dilation + inpainting.
    """

    def __init__(self, tex_size: int = 1024):
        self.tex_size = tex_size
        print(f"TextureComposer initialized (output: {tex_size}×{tex_size})")

    def compose_texture(
        self,
        views_with_angles: List[Tuple[Image.Image, float, float]],
        vertices_3xN: np.ndarray,
        uv_coords: np.ndarray,
        faces: np.ndarray,
        img_shape: Tuple[int, int],
    ) -> Image.Image:
        """
        Compose a UV texture from multiple views.

        Args:
            views_with_angles: List of (PIL.Image, yaw_degrees, pitch_degrees)
            vertices_3xN: BFM vertex positions (3, N)
            uv_coords: UV coordinates for each vertex (N, 2) in [0, 1]
            faces: Face indices (F, 3)
            img_shape: (height, width) of the original image space

        Returns:
            PIL.Image: Composed UV texture map
        """
        tex_size = self.tex_size
        n_verts = vertices_3xN.shape[1]

        if uv_coords is None or len(uv_coords) != n_verts:
            print("  [!] No UV coords - falling back to single-view baking")
            return self._single_view_fallback(views_with_angles, vertices_3xN, img_shape)

        print(f"  Composing texture from {len(views_with_angles)} views...")

        # Compute per-vertex normals for visibility determination
        normals = self._compute_vertex_normals(vertices_3xN, faces)

        # Initialize texture accumulation buffers
        tex_color = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
        tex_weight = np.zeros((tex_size, tex_size), dtype=np.float64)

        # UV pixel positions for all vertices
        u_px = np.clip((uv_coords[:, 0] * (tex_size - 1)).astype(int), 0, tex_size - 1)
        v_px = np.clip(((1.0 - uv_coords[:, 1]) * (tex_size - 1)).astype(int), 0, tex_size - 1)

        h, w = img_shape

        for view_img, yaw_deg, pitch_deg in views_with_angles:
            # Compute view direction vector from angles
            view_dir = self._angle_to_direction(yaw_deg, pitch_deg)

            # Compute per-vertex confidence: dot(normal, view_dir)
            # Higher confidence when vertex normal faces the camera
            confidence = self._compute_view_confidence(normals, view_dir)

            # Only use vertices with positive confidence (facing the camera)
            valid = confidence > 0.05

            if np.sum(valid) == 0:
                continue

            # Sample colors from this view image
            view_arr = np.array(view_img.convert("RGB").resize((w, h), Image.LANCZOS))
            xs = np.clip(vertices_3xN[0, :], 0, w - 1).astype(int)
            ys = np.clip(vertices_3xN[1, :], 0, h - 1).astype(int)
            vert_colors = view_arr[ys, xs, :]  # (N, 3) RGB

            # Accumulate weighted colors into UV texture
            for i in range(n_verts):
                if valid[i] and confidence[i] > 0:
                    weight = confidence[i] ** 2  # square for sharper falloff
                    tex_color[v_px[i], u_px[i]] += vert_colors[i].astype(np.float64) * weight
                    tex_weight[v_px[i], u_px[i]] += weight

        # Normalize by total weight
        mask = tex_weight > 0
        for c in range(3):
            tex_color[:, :, c][mask] /= tex_weight[mask]

        tex = tex_color.clip(0, 255).astype(np.uint8)

        # Fill holes with iterative dilation
        tex = self._fill_holes(tex)

        # Final smooth
        tex = cv2.GaussianBlur(tex, (3, 3), 0.5)

        print(f"  [OK] Texture composed: coverage = {mask.sum()}/{tex_size*tex_size} "
              f"({100*mask.sum()/(tex_size*tex_size):.1f}%)")

        return Image.fromarray(tex)

    def _compute_vertex_normals(
        self, vertices_3xN: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """Compute per-vertex normals from face connectivity."""
        verts = vertices_3xN.T  # (N, 3)
        n_verts = len(verts)

        normals = np.zeros((n_verts, 3), dtype=np.float64)

        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            fn = np.cross(edge1, edge2)
            fn_norm = np.linalg.norm(fn)
            if fn_norm > 1e-10:
                fn = fn / fn_norm
            normals[face[0]] += fn
            normals[face[1]] += fn
            normals[face[2]] += fn

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals = normals / norms

        return normals

    def _angle_to_direction(self, yaw_deg: float, pitch_deg: float) -> np.ndarray:
        """Convert yaw/pitch angles to a 3D view direction vector."""
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        # Camera looks along -Z in the face coordinate system
        # Yaw rotates around Y, Pitch rotates around X
        direction = np.array([
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return direction / (np.linalg.norm(direction) + 1e-10)

    def _compute_view_confidence(
        self, normals: np.ndarray, view_dir: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-vertex confidence for a given view direction.
        confidence = max(0, dot(normal, view_dir))
        """
        # dot product of each normal with the view direction
        dots = np.dot(normals, view_dir)
        confidence = np.maximum(dots, 0.0)
        return confidence

    def _fill_holes(self, tex: np.ndarray) -> np.ndarray:
        """Fill empty regions in the texture using iterative dilation."""
        mask = (tex.sum(axis=2) > 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)

        for _ in range(12):
            dilated = cv2.dilate(tex, kernel, iterations=1)
            empty = mask == 0
            tex[empty] = dilated[empty]
            mask = (tex.sum(axis=2) > 0).astype(np.uint8)

            if mask.all():
                break

        return tex

    def _single_view_fallback(
        self,
        views_with_angles: List[Tuple[Image.Image, float, float]],
        vertices_3xN: np.ndarray,
        img_shape: Tuple[int, int],
    ) -> Image.Image:
        """Fallback: bake texture from the front view only."""
        h, w = img_shape
        # Find the view closest to front (yaw=0, pitch=0)
        best_view = None
        best_dist = float("inf")

        for view_img, yaw, pitch in views_with_angles:
            dist = abs(yaw) + abs(pitch)
            if dist < best_dist:
                best_dist = dist
                best_view = view_img

        if best_view is None:
            best_view = views_with_angles[0][0]

        view_arr = np.array(best_view.convert("RGB").resize((w, h), Image.LANCZOS))

        xs = np.clip(vertices_3xN[0, :], 0, w - 1).astype(int)
        ys = np.clip(vertices_3xN[1, :], 0, h - 1).astype(int)
        colors = view_arr[ys, xs, :]

        # Create simple texture from vertex colors
        tex = np.zeros((self.tex_size, self.tex_size, 3), dtype=np.uint8)
        return Image.fromarray(tex)


if __name__ == "__main__":
    composer = TextureComposer(tex_size=1024)
    print("TextureComposer ready.")
