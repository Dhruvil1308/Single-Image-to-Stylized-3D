"""
FaceAnimator V3 — Realistic 2D face animation with full facial expressions.

Animates: mouth (lip sync), eyes (blinks + squint), brows (raise),
jaw (downward), cheeks (pull inward), and subtle head micro-motion.
Uses smooth displacement fields via cv2.remap for artifact-free output.
"""

import numpy as np
from PIL import Image
import cv2
import random

try:
    import mediapipe as mp
except ImportError:
    mp = None


# ── MediaPipe Face Mesh landmark indices ──
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 152
NOSE_TIP = 1
NOSE_BRIDGE = 6

# Eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Eyelid contours for realistic blink
LEFT_UPPER_EYELID = [159, 160, 161, 246, 33]
LEFT_LOWER_EYELID = [145, 144, 153, 154, 33]
RIGHT_UPPER_EYELID = [386, 387, 388, 466, 263]
RIGHT_LOWER_EYELID = [374, 373, 380, 381, 263]

# Brow landmarks
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]

# Cheek landmarks
LEFT_CHEEK = [116, 117, 118, 119, 100]
RIGHT_CHEEK = [345, 346, 347, 348, 329]

# Jaw contour
JAW_LINE = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]

# Forehead anchors (never move)
FOREHEAD = [10, 151, 9, 8, 168, 6, 197, 195, 5]


class FaceAnimator:
    """
    Creates realistic animated face frames with full facial expressions:
    - Lip sync (mouth opening/closing synchronized to audio)
    - Eye blinks (natural periodic blinking)
    - Eye squint (slight squint on loud syllables)
    - Brow raise (expressiveness on emphasis)
    - Jaw movement (lower face drops with mouth)
    - Cheek pull (inward pull when mouth opens wide)
    - Head micro-motion (subtle natural sway)
    """

    def __init__(self):
        if mp is None:
            raise ImportError("mediapipe is required: pip install mediapipe")

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.source_img = None
        self.landmarks = None
        self.img_h = 0
        self.img_w = 0
        print("FaceAnimator initialized (MediaPipe Face Mesh).")

    def set_source(self, pil_image):
        """Set the source face image and detect landmarks."""
        img_np = np.array(pil_image.convert("RGB"))
        self.img_h, self.img_w = img_np.shape[:2]
        self.source_img = img_np.copy()

        results = self.face_mesh.process(img_np)
        if not results.multi_face_landmarks:
            print("  [!] No face detected by MediaPipe")
            return False

        face_lm = results.multi_face_landmarks[0]
        self.landmarks = np.array(
            [(lm.x * self.img_w, lm.y * self.img_h) for lm in face_lm.landmark],
            dtype=np.float32,
        )

        self._compute_face_geometry()
        print(f"  [OK] Face mesh: {len(self.landmarks)} landmarks detected")
        return True

    def _compute_face_geometry(self):
        """Pre-compute facial geometry for animation."""
        lm = self.landmarks

        # Mouth geometry
        self.mouth_center = (lm[UPPER_LIP_CENTER] + lm[LOWER_LIP_CENTER]) / 2
        self.natural_mouth_h = np.linalg.norm(lm[UPPER_LIP_CENTER] - lm[LOWER_LIP_CENTER])
        self.mouth_width = np.linalg.norm(lm[MOUTH_LEFT] - lm[MOUTH_RIGHT])
        self.nose_to_chin = np.linalg.norm(lm[NOSE_TIP] - lm[CHIN])

        # Eye geometry
        self.left_eye_center = (lm[LEFT_EYE_TOP] + lm[LEFT_EYE_BOTTOM]) / 2
        self.right_eye_center = (lm[RIGHT_EYE_TOP] + lm[RIGHT_EYE_BOTTOM]) / 2
        self.left_eye_h = abs(lm[LEFT_EYE_TOP][1] - lm[LEFT_EYE_BOTTOM][1])
        self.right_eye_h = abs(lm[RIGHT_EYE_TOP][1] - lm[RIGHT_EYE_BOTTOM][1])
        self.left_eye_w = abs(lm[LEFT_EYE_RIGHT][0] - lm[LEFT_EYE_LEFT][0])
        self.right_eye_w = abs(lm[RIGHT_EYE_RIGHT][0] - lm[RIGHT_EYE_LEFT][0])

        # Face center
        self.face_center = (lm[NOSE_BRIDGE] + lm[CHIN]) / 2

        # Warp region for mouth (from nose to below chin)
        nose_y = lm[NOSE_TIP][1]
        chin_y = lm[CHIN][1]
        self.mouth_warp_top = nose_y + (chin_y - nose_y) * 0.15
        self.mouth_warp_bottom = min(chin_y + (chin_y - nose_y) * 0.35, self.img_h - 1)
        self.mouth_warp_left = max(lm[MOUTH_LEFT][0] - self.mouth_width * 0.7, 0)
        self.mouth_warp_right = min(lm[MOUTH_RIGHT][0] + self.mouth_width * 0.7, self.img_w - 1)

    def generate_frames(self, envelope, fps=30):
        """
        Generate animated face frames with full facial expressions.

        envelope: numpy array of mouth-open values (0.0-2.0)
        Returns: list of numpy arrays (RGB images)
        """
        if self.source_img is None or self.landmarks is None:
            raise ValueError("Call set_source() first.")

        # Pre-generate blink schedule
        blink_schedule = self._generate_blink_schedule(len(envelope), fps)

        frames = []
        for i, val in enumerate(envelope):
            blink_val = blink_schedule[i]
            frame = self._animate_frame(float(val), blink_val, i, len(envelope), fps)
            frames.append(frame)

        return frames

    def _generate_blink_schedule(self, n_frames, fps):
        """
        Generate natural eye blink pattern.
        Humans blink every 3-6 seconds, each blink lasts ~150ms (4-5 frames at 30fps).
        """
        schedule = np.zeros(n_frames, dtype=np.float32)
        frame = 0
        random.seed(42)  # reproducible

        while frame < n_frames:
            # Next blink in 2.5-5.5 seconds
            gap = int((2.5 + random.random() * 3.0) * fps)
            blink_start = frame + gap

            if blink_start >= n_frames:
                break

            # Blink duration: 4-6 frames (~130-200ms)
            blink_dur = random.randint(4, 6)

            for j in range(blink_dur):
                idx = blink_start + j
                if idx < n_frames:
                    # Bell curve for natural blink: eyes close then open
                    t = j / max(blink_dur - 1, 1)
                    # Sine curve: 0 -> 1 -> 0
                    schedule[idx] = np.sin(t * np.pi)

            frame = blink_start + blink_dur

        return schedule

    def _animate_frame(self, mouth_open, blink_val, frame_idx, total_frames, fps):
        """
        Create a single animated frame with all facial expressions combined.
        """
        h, w = self.img_h, self.img_w

        # Combined displacement fields
        disp_x = np.zeros((h, w), dtype=np.float32)
        disp_y = np.zeros((h, w), dtype=np.float32)

        # 1. MOUTH OPENING (lip sync)
        if mouth_open > 0.03:
            self._add_mouth_displacement(disp_x, disp_y, mouth_open)

        # 2. JAW DROP (lower face moves down with mouth)
        if mouth_open > 0.1:
            self._add_jaw_displacement(disp_y, mouth_open)

        # 3. CHEEK PULL (cheeks pull inward on wide mouth opening)
        if mouth_open > 0.5:
            self._add_cheek_displacement(disp_x, mouth_open)

        # 4. EYE BLINK
        if blink_val > 0.05:
            self._add_blink_displacement(disp_y, blink_val)

        # 5. EYE SQUINT (on loud syllables)
        if mouth_open > 0.8:
            squint = min((mouth_open - 0.8) * 0.5, 0.4)
            self._add_eye_squint(disp_y, squint)

        # 6. BROW RAISE (on emphasis / loud parts)
        if mouth_open > 0.6:
            brow_amount = min((mouth_open - 0.6) * 0.4, 0.5)
            self._add_brow_displacement(disp_y, brow_amount)

        # 7. SUBTLE HEAD MICRO-MOTION (natural sway)
        self._add_head_motion(disp_x, disp_y, frame_idx, total_frames, fps, mouth_open)

        # Apply Gaussian blur for ultra-smooth blending of all effects
        k = max(int(self.natural_mouth_h * 0.4) | 1, 5)
        disp_x = cv2.GaussianBlur(disp_x, (k, k), 0)
        disp_y = cv2.GaussianBlur(disp_y, (k, k), 0)

        # Build remap coordinates (inverse mapping)
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            map_x[y, :] = np.arange(w, dtype=np.float32) - disp_x[y, :]
            map_y[y, :] = float(y) - disp_y[y, :]

        # Warp with high-quality interpolation
        warped = cv2.remap(
            self.source_img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return warped

    # ──────────────────────────────────────────────
    #  Displacement generators for each expression
    # ──────────────────────────────────────────────

    def _add_mouth_displacement(self, disp_x, disp_y, mouth_open):
        """Lip sync: lower lip drops, upper lip rises slightly."""
        lm = self.landmarks
        mouth_open = min(mouth_open, 2.5)

        max_disp = mouth_open * self.natural_mouth_h * 0.38
        upper_lip_y = lm[UPPER_LIP_CENTER][1]
        lower_lip_y = lm[LOWER_LIP_CENTER][1]
        chin_y = lm[CHIN][1]
        cx = self.mouth_center[0]

        y_start = int(self.mouth_warp_top)
        y_end = int(self.mouth_warp_bottom) + 1
        x_start = int(self.mouth_warp_left)
        x_end = int(self.mouth_warp_right) + 1

        for y_px in range(y_start, min(y_end, self.img_h)):
            y = float(y_px)

            # Vertical profile
            if y <= upper_lip_y:
                # Above upper lip: slight upward pull
                t = max(0, 1.0 - (upper_lip_y - y) / max(self.natural_mouth_h * 1.5, 1))
                vf = -0.12 * t * t  # quadratic ease-in upward
            elif y <= lower_lip_y:
                # Between lips: gradual open
                t = (y - upper_lip_y) / max(lower_lip_y - upper_lip_y, 1)
                vf = t * 1.0
            elif y <= chin_y:
                # Lower lip to chin: strong push down fading
                t = (y - lower_lip_y) / max(chin_y - lower_lip_y, 1)
                vf = 1.0 - t * 0.5
            else:
                # Below chin: fade out
                t = (y - chin_y) / max(self.mouth_warp_bottom - chin_y, 1)
                vf = 0.5 * max(0, 1.0 - t)

            for x_px in range(x_start, min(x_end, self.img_w)):
                x = float(x_px)
                dx = abs(x - cx) / max(self.mouth_width * 0.8, 1)
                if dx < 1.0:
                    hf = 0.5 * (1.0 + np.cos(np.pi * dx))
                else:
                    hf = 0.0

                disp_y[y_px, x_px] += max_disp * vf * hf

                # Slight horizontal pull toward center (mouth corners tighten)
                if y >= upper_lip_y and y <= chin_y and dx < 0.9:
                    pull = mouth_open * 0.4 * hf * (1.0 if x < cx else -1.0)
                    if y >= lower_lip_y:
                        pull *= 0.5
                    disp_x[y_px, x_px] += pull

    def _add_jaw_displacement(self, disp_y, mouth_open):
        """Jaw drop: entire lower face shifts down subtly."""
        lm = self.landmarks
        jaw_disp = mouth_open * self.natural_mouth_h * 0.08  # subtle

        lower_lip_y = lm[LOWER_LIP_CENTER][1]
        chin_y = lm[CHIN][1]
        cx = self.face_center[0]
        face_w = self.mouth_width * 1.5

        for y_px in range(int(lower_lip_y), int(min(self.mouth_warp_bottom, self.img_h))):
            y = float(y_px)
            if y <= chin_y:
                t = (y - lower_lip_y) / max(chin_y - lower_lip_y, 1)
                vf = t * 0.7 + 0.3
            else:
                t = (y - chin_y) / max(self.mouth_warp_bottom - chin_y, 1)
                vf = max(0, 1.0 - t)

            for x_px in range(int(self.mouth_warp_left), int(min(self.mouth_warp_right, self.img_w))):
                x = float(x_px)
                dx = abs(x - cx) / max(face_w, 1)
                hf = max(0, 1.0 - dx * dx)
                disp_y[y_px, x_px] += jaw_disp * vf * hf

    def _add_cheek_displacement(self, disp_x, mouth_open):
        """Cheeks pull slightly inward when mouth opens wide."""
        lm = self.landmarks
        pull = min((mouth_open - 0.5) * 0.6, 1.0) * 1.5  # pixels

        for side, cheek_ids in [("left", LEFT_CHEEK), ("right", RIGHT_CHEEK)]:
            pts = lm[cheek_ids]
            cy = np.mean(pts[:, 1])
            cx = np.mean(pts[:, 0])
            radius = self.mouth_width * 0.4

            direction = 1.0 if side == "left" else -1.0  # pull toward center

            y_start = max(0, int(cy - radius))
            y_end = min(self.img_h, int(cy + radius))
            x_start = max(0, int(cx - radius))
            x_end = min(self.img_w, int(cx + radius))

            for y_px in range(y_start, y_end):
                for x_px in range(x_start, x_end):
                    dist = np.sqrt((y_px - cy) ** 2 + (x_px - cx) ** 2)
                    if dist < radius:
                        factor = 0.5 * (1.0 + np.cos(np.pi * dist / radius))
                        disp_x[y_px, x_px] += pull * factor * direction

    def _add_blink_displacement(self, disp_y, blink_val):
        """Eye blink: upper eyelid moves down, lower eyelid moves up."""
        for eye_center, eye_h, eye_w, upper_ids, lower_ids in [
            (self.left_eye_center, self.left_eye_h, self.left_eye_w,
             LEFT_UPPER_EYELID, LEFT_LOWER_EYELID),
            (self.right_eye_center, self.right_eye_h, self.right_eye_w,
             RIGHT_UPPER_EYELID, RIGHT_LOWER_EYELID),
        ]:
            cy, cx = eye_center[1], eye_center[0]
            # Blink displacement: close the eye gap
            close_amount = blink_val * eye_h * 0.9  # how much to close

            radius_y = eye_h * 1.5
            radius_x = eye_w * 0.8

            y_start = max(0, int(cy - radius_y))
            y_end = min(self.img_h, int(cy + radius_y))
            x_start = max(0, int(cx - radius_x))
            x_end = min(self.img_w, int(cx + radius_x))

            for y_px in range(y_start, y_end):
                for x_px in range(x_start, x_end):
                    dy = (y_px - cy) / max(radius_y, 1)
                    dx = (x_px - cx) / max(radius_x, 1)
                    dist = np.sqrt(dy * dy + dx * dx)

                    if dist < 1.0:
                        factor = 0.5 * (1.0 + np.cos(np.pi * dist))

                        if y_px < cy:
                            # Upper eyelid: push down
                            disp_y[y_px, x_px] += close_amount * factor * 0.7
                        else:
                            # Lower eyelid: push up
                            disp_y[y_px, x_px] -= close_amount * factor * 0.3

    def _add_eye_squint(self, disp_y, squint_amount):
        """Slight eye squint on loud syllables (natural speech expression)."""
        for eye_center, eye_h, eye_w in [
            (self.left_eye_center, self.left_eye_h, self.left_eye_w),
            (self.right_eye_center, self.right_eye_h, self.right_eye_w),
        ]:
            cy, cx = eye_center[1], eye_center[0]
            squeeze = squint_amount * eye_h * 0.4

            radius_y = eye_h * 1.2
            radius_x = eye_w * 0.7

            y_start = max(0, int(cy - radius_y))
            y_end = min(self.img_h, int(cy + radius_y))
            x_start = max(0, int(cx - radius_x))
            x_end = min(self.img_w, int(cx + radius_x))

            for y_px in range(y_start, y_end):
                for x_px in range(x_start, x_end):
                    dy = (y_px - cy) / max(radius_y, 1)
                    dx = (x_px - cx) / max(radius_x, 1)
                    dist = np.sqrt(dy * dy + dx * dx)

                    if dist < 1.0:
                        factor = 0.5 * (1.0 + np.cos(np.pi * dist))
                        if y_px < cy:
                            disp_y[y_px, x_px] += squeeze * factor
                        else:
                            disp_y[y_px, x_px] -= squeeze * factor * 0.5

    def _add_brow_displacement(self, disp_y, amount):
        """Brow raise on emphatic speech."""
        lm = self.landmarks
        all_brow = LEFT_BROW + RIGHT_BROW
        brow_pts = lm[all_brow]

        brow_cy = np.mean(brow_pts[:, 1])
        brow_cx = np.mean(brow_pts[:, 0])
        brow_w = np.ptp(brow_pts[:, 0]) * 0.6
        brow_h = max(np.ptp(brow_pts[:, 1]) * 1.5, 10)

        raise_px = amount * 3.0

        y_start = max(0, int(brow_cy - brow_h))
        y_end = min(self.img_h, int(brow_cy + brow_h * 0.5))
        x_start = max(0, int(brow_cx - brow_w))
        x_end = min(self.img_w, int(brow_cx + brow_w))

        for y_px in range(y_start, y_end):
            for x_px in range(x_start, x_end):
                dy = (y_px - brow_cy) / max(brow_h, 1)
                dx = (x_px - brow_cx) / max(brow_w, 1)
                dist = np.sqrt(dy * dy + dx * dx)
                if dist < 1.0:
                    factor = 0.5 * (1.0 + np.cos(np.pi * dist))
                    disp_y[y_px, x_px] -= raise_px * factor  # upward

    def _add_head_motion(self, disp_x, disp_y, frame_idx, total_frames, fps, mouth_open):
        """
        Subtle head micro-motion — natural sway while speaking.
        Uses slow sine waves for organic movement.
        """
        t = frame_idx / fps  # time in seconds

        # Very subtle nodding (vertical) — tied to speech
        nod = np.sin(t * 2.5) * 0.3 * min(mouth_open, 1.0)

        # Very subtle swaying (horizontal)
        sway = np.sin(t * 1.3) * 0.2

        if abs(nod) > 0.01 or abs(sway) > 0.01:
            # Apply uniform displacement to entire face region
            face_top = int(self.landmarks[FOREHEAD[0]][1])
            face_bot = int(min(self.landmarks[CHIN][1] + 20, self.img_h))

            # Gentle gradient: more motion at top, less at bottom (pivot at chin)
            for y_px in range(face_top, face_bot):
                lever = 1.0 - (y_px - face_top) / max(face_bot - face_top, 1)
                disp_x[y_px, :] += sway * lever
                disp_y[y_px, :] += nod * lever
