"""
FaceReconstructor — 3DDFA_V2 ONNX-based 3D face reconstruction.

Produces a dense BFM mesh (~38K vertices, ~76K faces) with UV-mapped
face texture from a single photograph. Supports parametric shape and
expression modification via 3DMM coefficients.
"""

import os
import sys
import time
import numpy as np
import cv2
import trimesh
import trimesh.visual
import trimesh.visual.material
import scipy.io as sio
import onnxruntime
from PIL import Image

# ── Path setup for 3DDFA_V2 libs ──
TDDFA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "3DDFA_V2")
TDDFA_DIR = os.path.abspath(TDDFA_DIR)
if TDDFA_DIR not in sys.path:
    sys.path.insert(0, TDDFA_DIR)

from utils.io import _load
from utils.functions import crop_img, parse_roi_box_from_bbox
from utils.tddfa_util import _parse_param, similar_transform, _to_ctype
from bfm.bfm import BFMModel
from bfm.bfm_onnx import convert_bfm_to_onnx
from utils.onnx import convert_to_onnx

_abs = lambda fn: os.path.join(TDDFA_DIR, fn)


class FaceReconstructor:
    """
    3D Face Reconstruction using 3DDFA_V2 + BFM morphable model.

    Pipeline:
      1. Detect face bounding box (via Mediapipe or cv2)
      2. Crop & normalise → ONNX MobileNet → 62-dim 3DMM params
      3. Decompose into: 12 pose + 40 shape + 10 expression
      4. Reconstruct dense BFM mesh (38,365 vertices, 76,073 faces)
      5. Sample face texture from the input image
      6. Build UV-textured trimesh for GLB export
    """

    def __init__(self):
        t0 = time.time()
        cfg = self._load_config()

        # ── BFM morphable model ──
        bfm_fp = _abs(cfg["bfm_fp"])
        self.bfm = BFMModel(bfm_fp, shape_dim=40, exp_dim=10)
        self.tri = self.bfm.tri  # (n_faces, 3)

        # ── ONNX sessions ──
        # 1) MobileNet regressor (image → 62 params)
        checkpoint_fp = _abs(cfg["checkpoint_fp"])
        onnx_fp = checkpoint_fp.replace(".pth", ".onnx")
        if not os.path.exists(onnx_fp):
            print("  Converting checkpoint to ONNX (one-time)...")
            convert_to_onnx(
                checkpoint_fp=checkpoint_fp,
                onnx_fp=onnx_fp,
                arch=cfg["arch"],
                num_params=cfg["num_params"],
                widen_factor=cfg.get("widen_factor", 1),
                size=cfg["size"],
                mode="onnx"
            )
        self.session = onnxruntime.InferenceSession(onnx_fp, None)

        # 2) BFM dense reconstruction session
        bfm_onnx_fp = bfm_fp.replace(".pkl", ".onnx")
        if not os.path.exists(bfm_onnx_fp):
            convert_bfm_to_onnx(bfm_onnx_fp, shape_dim=40, exp_dim=10)
        self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_fp, None)

        # ── Normalisation stats ──
        stats_fp = _abs(f"configs/param_mean_std_62d_{cfg['size']}x{cfg['size']}.pkl")
        stats = _load(stats_fp)
        self.param_mean = stats["mean"]
        self.param_std = stats["std"]
        self.size = cfg["size"]

        # ── UV coordinates ──
        uv_mat_fp = _abs("configs/BFM_UV.mat")
        if os.path.exists(uv_mat_fp):
            uv_all = sio.loadmat(uv_mat_fp)["UV"].astype(np.float32)
            indices_fp = _abs("configs/indices.npy")
            if os.path.exists(indices_fp):
                indices = np.load(indices_fp)
                self.uv_coords = uv_all[indices, :]
            else:
                self.uv_coords = uv_all
        else:
            self.uv_coords = None

        # ── Face detector (lightweight cv2 cascade) ──
        cascade_fp = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_fp)

        # ── Current parameters (for slider modification) ──
        self.current_param = None
        self.current_roi_box = None
        self.current_img = None

        dt = time.time() - t0
        print(f"FaceReconstructor initialized in {dt:.1f}s (BFM + ONNX)")

    # ──────────────────────────────────────
    #  Config
    # ──────────────────────────────────────
    def _load_config(self):
        import yaml
        cfg_fp = _abs("configs/mb1_120x120.yml")
        with open(cfg_fp) as f:
            return yaml.safe_load(f)

    # ──────────────────────────────────────
    #  Face detection
    # ──────────────────────────────────────
    def _detect_face(self, img_bgr):
        """Detect face bbox using cv2 cascade."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            # Fallback: assume entire image is the face
            h, w = img_bgr.shape[:2]
            margin = int(min(h, w) * 0.05)
            return [margin, margin, w - margin, h - margin]
        x, y, w, h = faces[0]
        return [x, y, x + w, y + h]

    # ──────────────────────────────────────
    #  3DMM param regression
    # ──────────────────────────────────────
    def _predict_params(self, img_bgr, bbox):
        """Run ONNX MobileNet to predict 62 3DMM parameters."""
        roi_box = parse_roi_box_from_bbox(bbox)
        img = crop_img(img_bgr, roi_box)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img_input = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        img_input = (img_input - 127.5) / 128.0

        param = self.session.run(None, {"input": img_input})[0]
        param = param.flatten().astype(np.float32)
        param = param * self.param_std + self.param_mean
        return param, roi_box

    # ──────────────────────────────────────
    #  Dense mesh reconstruction
    # ──────────────────────────────────────
    def _reconstruct_vertices(self, param, roi_box):
        """Reconstruct dense 3D vertices from 3DMM parameters."""
        R, offset, alpha_shp, alpha_exp = _parse_param(param)

        # Dense BFM reconstruction via ONNX
        inp = {
            "R": R, "offset": offset,
            "alpha_shp": alpha_shp, "alpha_exp": alpha_exp
        }
        pts3d = self.bfm_session.run(None, inp)[0]
        pts3d = similar_transform(pts3d, roi_box, self.size)
        return pts3d  # shape: (3, n_vertices)

    # ──────────────────────────────────────
    #  Texture baking (no Sim3DR needed)
    # ──────────────────────────────────────
    def _bake_texture(self, img_bgr, vertices_3xN, tex_size=1024):
        """
        Create a UV texture by sampling pixel colours from the image
        at each vertex position, then painting them into UV space.
        """
        h, w = img_bgr.shape[:2]
        n_verts = vertices_3xN.shape[1]

        if self.uv_coords is None or len(self.uv_coords) != n_verts:
            # Fallback: vertex colors
            return None, None

        # Sample vertex colours from image
        xs = np.clip(vertices_3xN[0, :], 0, w - 1)
        ys = np.clip(vertices_3xN[1, :], 0, h - 1)
        xi = np.round(xs).astype(np.int32)
        yi = np.round(ys).astype(np.int32)
        vert_colors = img_bgr[yi, xi, :]  # (n_verts, 3) BGR

        # Build UV texture image
        tex = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
        uv = self.uv_coords.copy()
        u_px = np.clip((uv[:, 0] * (tex_size - 1)).astype(int), 0, tex_size - 1)
        v_px = np.clip(((1.0 - uv[:, 1]) * (tex_size - 1)).astype(int), 0, tex_size - 1)

        # Paint each vertex colour into UV space
        tex[v_px, u_px] = vert_colors[:, ::-1]  # BGR → RGB

        # Fill holes with dilation
        mask = (tex.sum(axis=2) > 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        for _ in range(8):
            dilated = cv2.dilate(tex, kernel, iterations=1)
            empty = mask == 0
            tex[empty] = dilated[empty]
            mask = (tex.sum(axis=2) > 0).astype(np.uint8)

        # Gaussian blur for smooth texture
        tex = cv2.GaussianBlur(tex, (3, 3), 0)

        tex_pil = Image.fromarray(tex)
        return tex_pil, uv

    # ──────────────────────────────────────
    #  Build textured trimesh
    # ──────────────────────────────────────
    def _build_trimesh(self, vertices_3xN, img_bgr):
        """Build a trimesh.Trimesh with UV texture from BFM mesh."""
        h, w = img_bgr.shape[:2]

        # Vertices: (3, N) → (N, 3), flip Y for 3D viewer
        verts = vertices_3xN.T.copy()  # (N, 3)
        verts[:, 1] = h - verts[:, 1]  # flip Y
        # Centre at origin
        centroid = verts.mean(axis=0)
        verts -= centroid
        # Scale to reasonable size
        scale = np.max(np.abs(verts)) + 1e-6
        verts = verts / scale * 120.0

        faces = self.tri.copy()

        # Try UV texture baking
        tex_img, uv = self._bake_texture(img_bgr, vertices_3xN)

        if tex_img is not None and uv is not None:
            # UV-textured mesh
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex_img,
                metallicFactor=0.0,
                roughnessFactor=0.7
            )
            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                process=False
            )
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv[:len(verts)],
                material=material
            )
        else:
            # Fallback: vertex colors
            xs = np.clip(vertices_3xN[0, :], 0, w - 1).astype(int)
            ys = np.clip(vertices_3xN[1, :], 0, h - 1).astype(int)
            colors = img_bgr[ys, xs, ::-1]  # BGR→RGB
            colors_rgba = np.hstack([colors, np.full((len(colors), 1), 255, dtype=np.uint8)])
            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_colors=colors_rgba,
                process=False
            )

        return mesh

    # ──────────────────────────────────────
    #  Public API: Full reconstruction
    # ──────────────────────────────────────
    def reconstruct(self, pil_image):
        """
        Reconstruct a 3D face mesh from a PIL image.
        Returns: trimesh.Trimesh with UV texture
        """
        t0 = time.time()

        # Convert PIL → BGR numpy
        img_rgb = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Resize if too large
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1024:
            ratio = 1024 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * ratio), int(h * ratio)))

        # 1. Detect face
        print("  [1/4] Detecting face...")
        bbox = self._detect_face(img_bgr)

        # 2. Predict 3DMM params
        print("  [2/4] Regressing 3DMM parameters...")
        param, roi_box = self._predict_params(img_bgr, bbox)
        self.current_param = param.copy()
        self.current_roi_box = roi_box
        self.current_img = img_bgr.copy()

        # 3. Reconstruct dense mesh
        print("  [3/4] Reconstructing dense BFM mesh...")
        vertices = self._reconstruct_vertices(param, roi_box)
        n_verts = vertices.shape[1]
        n_faces = len(self.tri)
        print(f"        Vertices: {n_verts:,} | Faces: {n_faces:,}")

        # 4. Build textured mesh
        print("  [4/4] Baking UV texture...")
        mesh = self._build_trimesh(vertices, img_bgr)

        dt = time.time() - t0
        print(f"  ✅ 3D reconstruction complete in {dt:.1f}s")
        return mesh

    # ──────────────────────────────────────
    #  Public API: Modify parameters
    # ──────────────────────────────────────
    def modify_params(self, shape_deltas=None, expr_deltas=None):
        """
        Modify shape/expression parameters and regenerate the mesh.

        shape_deltas: dict of {index: value} for BFM shape params (0-39)
        expr_deltas: dict of {index: value} for BFM expression params (0-9)

        Shape param indices (approximate semantic meaning):
          0: Overall face size/scale
          1: Face width/fullness
          2: Jaw width
          3: Nose bridge height
          4: Forehead height
          5-39: Other face shape variations

        Expression param indices:
          0: Brow raise
          1: Smile (mouth corners)
          2: Mouth open
          3: Eye squint
          4-9: Other expressions
        """
        if self.current_param is None:
            print("No previous reconstruction. Call reconstruct() first.")
            return None

        param = self.current_param.copy()
        R, offset, alpha_shp, alpha_exp = _parse_param(param)

        # Modify shape (indices 12-51 in the 62-dim param vector)
        if shape_deltas:
            for idx, val in shape_deltas.items():
                if 0 <= idx < 40:
                    alpha_shp[idx, 0] += val * 10.0  # Scale for visible effect

        # Modify expression (indices 52-61)
        if expr_deltas:
            for idx, val in expr_deltas.items():
                if 0 <= idx < 10:
                    alpha_exp[idx, 0] += val * 5.0

        # Reconstruct with modified params
        inp = {
            "R": R, "offset": offset,
            "alpha_shp": alpha_shp, "alpha_exp": alpha_exp
        }
        vertices = self.bfm_session.run(None, inp)[0]
        vertices = similar_transform(vertices, self.current_roi_box, self.size)

        mesh = self._build_trimesh(vertices, self.current_img)
        return mesh


if __name__ == "__main__":
    recon = FaceReconstructor()
    print(f"BFM faces: {len(recon.tri)}")
    print("FaceReconstructor ready.")
