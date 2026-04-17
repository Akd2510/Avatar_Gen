import cv2
import numpy as np


class HeadBlender:
    def inpaint_template(self, template_img, template_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated_mask = cv2.dilate(template_mask, kernel, iterations=1)
        return cv2.inpaint(template_img, dilated_mask, 3, cv2.INPAINT_TELEA)
