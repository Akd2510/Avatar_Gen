import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self, model_path="face_landmarker.task"):
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise

    def get_landmarks(self, image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = self.detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return None
        h, w, _ = image.shape
        landmarks = detection_result.face_landmarks[0]
        return np.array([(int(pt.x * w), int(pt.y * h)) for pt in landmarks])
