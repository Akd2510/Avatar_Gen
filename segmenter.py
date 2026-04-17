import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HeadSegmenter:
    def __init__(self, model_path="selfie_segmenter.tflite"):
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
        except Exception as e:
            raise

    def get_head_mask(self, image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = self.segmenter.segment(mp_image)
        person_mask = result.confidence_masks[
            1 if len(result.confidence_masks) > 1 else 0
        ].numpy_view()
        binary_mask = np.where(person_mask > 0.5, 255, 0).astype(np.uint8)
        if binary_mask[0, 0] == 255:
            binary_mask = cv2.bitwise_not(binary_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
        return cv2.GaussianBlur(binary_mask, (5, 5), 0)
