import cv2
import numpy as np


class MaskGenerator:
    def __init__(self):
        # Core facial features only
        self.feature_indices = [
            # Eyes & Eyebrows
            33,
            133,
            362,
            263,
            70,
            63,
            105,
            66,
            107,
            336,
            296,
            334,
            293,
            300,
            # Nose
            168,
            6,
            197,
            195,
            5,
            4,
            1,
            19,
            94,
            # Mouth
            61,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            321,
            375,
            291,
            409,
            270,
            269,
            267,
            0,
            37,
        ]

    def create_mask(self, image_shape, landmarks):
        """Generates a smooth, rounded binary mask around the inner facial features."""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Gather points
        feature_pts = np.array(
            [landmarks[idx] for idx in self.feature_indices], dtype=np.int32
        )

        # 2. Draw a tight boundary (Convex Hull)
        hull = cv2.convexHull(feature_pts)
        cv2.fillConvexPoly(mask, hull, 255)

        # 3. CRITICAL FIX: Round out the sharp corners!
        # We dilate the mask with a large circular kernel to create a smooth, oval-like boundary
        # that sits on the flat skin of the cheeks and forehead, removing angular lines.
        # Note: (35, 35) is a good starting size. If the mask bleeds into the hair, lower it to (25, 25).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        smooth_mask = cv2.dilate(mask, kernel, iterations=1)

        return smooth_mask
