import cv2
import numpy as np

class ImageBlender:
    def blend(self, warped_face, template_img, warped_mask):
        # 1. Force a clean binary mask
        _, binary_mask = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)
        
        # 2. Find the center
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return template_img
        M = cv2.moments(contours[0])
        if M["m00"] == 0: return template_img
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 3. Blending
        # MIXED_CLONE is better for keeping your glasses and sharp eyes
        return cv2.seamlessClone(warped_face, template_img, binary_mask, center, cv2.NORMAL_CLONE)