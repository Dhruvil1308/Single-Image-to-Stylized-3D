import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

class FaceExtractor:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    def detect_face(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        
        # Get the first face detected
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, ic = image.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
               int(bboxC.width * iw), int(bboxC.height * ih)
        
        return bbox

    def align_and_crop(self, image_path, output_size=(512, 512)):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        ih, iw, _ = image.shape
        
        # Key points for alignment: eyes
        # MediaPipe Face Mesh landmark indices:
        # Left eye center: 468, Right eye center: 473 (approx)
        # Using more robust indices from Mesh
        left_eye = np.array([landmarks.landmark[33].x * iw, landmarks.landmark[33].y * ih])
        right_eye = np.array([landmarks.landmark[263].x * iw, landmarks.landmark[263].y * ih])
        
        # Calculate angle for rotation
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Eye center
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1)
        rotated = cv2.warpAffine(image, M, (iw, ih), flags=cv2.INTER_CUBIC)
        
        # Recalculate landmarks on rotated image for better cropping
        rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        results_rot = self.face_mesh.process(rotated_rgb)
        
        if not results_rot.multi_face_landmarks:
            return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)).resize(output_size)

        landmarks_rot = results_rot.multi_face_landmarks[0]
        pts = np.array([(l.x * iw, l.y * ih) for l in landmarks_rot.landmark])
        
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        
        # Add padding
        w = xmax - xmin
        h = ymax - ymin
        size = max(w, h) * 1.5
        
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        x1 = max(0, int(cx - size / 2))
        y1 = max(0, int(cy - size / 2))
        x2 = min(iw, int(cx + size / 2))
        y2 = min(ih, int(cy + size / 2))
        
        cropped = rotated[y1:y2, x1:x2]
        final_image = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)
        
        return Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    # Test stub
    extractor = FaceExtractor()
    print("FaceExtractor initialized.")
