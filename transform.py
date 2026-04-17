import cv2
import numpy as np


class HeadTransformer:
    def align_head(self, src_img, src_mask, src_landmarks, tgt_img, tgt_landmarks):
        h_tgt, w_tgt = tgt_img.shape[:2]
        src_pts = np.array(
            [src_landmarks[33], src_landmarks[263], src_landmarks[152]],
            dtype=np.float32,
        )
        tgt_pts = np.array(
            [tgt_landmarks[33], tgt_landmarks[263], tgt_landmarks[152]],
            dtype=np.float32,
        )
        M, _ = cv2.estimateAffinePartial2D(src_pts, tgt_pts)
        if M is None:
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        aligned_img = cv2.warpAffine(
            src_img,
            M,
            (w_tgt, h_tgt),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        aligned_mask = cv2.warpAffine(
            src_mask,
            M,
            (w_tgt, h_tgt),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return aligned_img, aligned_mask
