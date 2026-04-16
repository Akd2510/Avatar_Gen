import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class FaceDetector:
    def __init__(self, model_asset_path='face_landmarker.task'):
        try:
            base_options = python.BaseOptions(model_asset_path=model_asset_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing Face Landmarker: {e}")
            raise

    def get_landmarks(self, image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            print("REJECTED: No face detected in the image.")
            return None

        landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        
        facial_pts = np.array([
            (int(pt.x * w), int(pt.y * h)) for pt in landmarks
        ])

        # --- THE GATE: Strict Side-Face Rejection ---
        nose_tip = facial_pts[1] 
        left_cheek = facial_pts[234]  # Extreme left edge of face
        right_cheek = facial_pts[454] # Extreme right edge of face
        
        # Calculate pixel distances
        dist_left = np.linalg.norm(nose_tip - left_cheek)
        dist_right = np.linalg.norm(nose_tip - right_cheek)
        
        # Calculate ratio (A perfectly straight face is 1.0)
        ratio = dist_left / (dist_right + 1e-6)
        
        # Strict tolerance bounds (0.8 to 1.25)
        if ratio < 0.80 or ratio > 1.25:
            print(f"GATE TRIGGERED: Side face detected (Ratio: {ratio:.2f}).")
            print("Please upload a photo looking directly at the camera.")
            return None
        # --------------------------------------------

        return facial_pts