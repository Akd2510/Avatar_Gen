import cv2
import numpy as np
from segmenter import HeadSegmenter

from blender import HeadBlender
from detector import FaceDetector
from transform import HeadTransformer


def apply_final_skin_filter(final_img, template_img, landmarks, intensity=0.15):
    face_indices = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
    face_pts = np.array([landmarks[idx] for idx in face_indices], dtype=np.int32)
    skin_mask = np.zeros(final_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(skin_mask, [face_pts], 255)
    safe_mask = cv2.erode(skin_mask, np.ones((15, 15), np.uint8))
    if not np.any(safe_mask):
        return final_img
    template_skin_color = cv2.mean(template_img, mask=safe_mask)[:3]
    color_wash = np.full_like(final_img, template_skin_color, dtype=np.uint8)
    filtered_img = cv2.addWeighted(final_img, 1.0 - intensity, color_wash, intensity, 0)
    alpha = (
        cv2.GaussianBlur(skin_mask, (51, 51), 0).astype(np.float32)[..., np.newaxis]
        / 255.0
    )
    return np.clip(filtered_img * alpha + final_img * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )


def remove_white_fringe(source_img, aln_mask, threshold=80):
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.subtract(
        cv2.dilate(aln_mask, kernel, iterations=2), cv2.erode(aln_mask, kernel)
    )
    near_white = (
        (source_img[..., 0] > 255 - threshold)
        & (source_img[..., 1] > 255 - threshold)
        & (source_img[..., 2] > 255 - threshold)
    ) & (boundary > 0)
    if not np.any(near_white):
        return source_img
    return cv2.inpaint(
        source_img, near_white.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA
    )


def add_drop_shadow(
    template_img, aligned_mask, landmarks, offset_y=10, blur=21, shadow_intensity=0.35
):
    h, w = aligned_mask.shape
    M = np.float32([[1, 0, 0], [0, 1, offset_y]])
    shifted_mask = cv2.warpAffine(aligned_mask, M, (w, h))
    shadow_area = cv2.subtract(shifted_mask, aligned_mask)
    left_jaw, right_jaw = landmarks[136][1], landmarks[365][1]
    if 0 < min(left_jaw, right_jaw) < h:
        shadow_area[: int(min(left_jaw, right_jaw)), :] = 0
    shadow_area = cv2.GaussianBlur(shadow_area, (blur, blur), 0)
    shadow_area[aligned_mask > 20] = 0
    shadow_alpha = (shadow_area / 255.0)[..., np.newaxis]
    return (template_img * (1.0 - shadow_alpha * shadow_intensity)).astype(np.uint8)


def prep_target_inpainting_mask(mask, landmarks):
    refined = mask.copy()
    chin_y = landmarks[152][1]
    if int(chin_y) < refined.shape[0]:
        refined[int(chin_y) :, :] = 0
    return refined


def clean_aligned_mask(aligned_mask, tgt_landmarks):
    refined = aligned_mask.copy()
    h, w = refined.shape
    l_jaw, r_jaw, chin = tgt_landmarks[136], tgt_landmarks[365], tgt_landmarks[152]
    neck_join_y = chin[1] + 35
    deletion_zone = np.zeros_like(refined)
    poly_pts = [
        (0, h),
        (0, l_jaw[1] + 50),
        (l_jaw[0] - 20, l_jaw[1] + 35),
        (chin[0], neck_join_y),
        (r_jaw[0] + 20, r_jaw[1] + 35),
        (w, r_jaw[1] + 50),
        (w, h),
    ]
    cv2.fillPoly(deletion_zone, [np.array(poly_pts, dtype=np.int32)], 255)
    feather = cv2.GaussianBlur(deletion_zone, (45, 45), 0).astype(np.float32) / 255.0
    refined_float = refined.astype(np.float32)
    refined_float = refined_float * (1.0 - feather)
    shoulder_shield = np.zeros_like(refined)
    cv2.rectangle(
        shoulder_shield, (0, int(l_jaw[1] + 20)), (int(l_jaw[0] - 10), h), 255, -1
    )
    cv2.rectangle(
        shoulder_shield, (int(r_jaw[0] + 10), int(r_jaw[1] + 20)), (w, h), 255, -1
    )
    refined_float[shoulder_shield > 0] = 0
    return refined_float.astype(np.uint8)


def match_skin_tone_smart(aligned_head, template_img, landmarks):
    face_indices = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
    face_pts = np.array([landmarks[idx] for idx in face_indices], dtype=np.int32)
    skin_mask = np.zeros(aligned_head.shape[:2], dtype=np.uint8)
    cv2.fillPoly(skin_mask, [face_pts], 255)
    safe_idx = cv2.erode(skin_mask, np.ones((11, 11), np.uint8)) > 0
    if not np.any(safe_idx):
        return aligned_head
    src_lab = cv2.cvtColor(aligned_head, cv2.COLOR_BGR2Lab).astype(np.float32)
    tgt_lab = cv2.cvtColor(template_img, cv2.COLOR_BGR2Lab).astype(np.float32)
    for i in range(3):
        m_s, m_t = (
            np.mean(src_lab[..., i][safe_idx]),
            np.mean(tgt_lab[..., i][safe_idx]),
        )
        src_lab[..., i] = src_lab[..., i] + (m_t - m_s)
    res_bgr = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
    noise = np.random.normal(0, 1.0, res_bgr.shape).astype(np.float32)
    res_bgr = np.clip(res_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    res_bgr = cv2.addWeighted(res_bgr, 0.90, aligned_head, 0.10, 0)
    alpha = cv2.cvtColor(
        cv2.GaussianBlur(skin_mask, (51, 51), 0).astype(np.float32) / 255.0,
        cv2.COLOR_GRAY2BGR,
    )
    return np.clip(
        res_bgr * alpha + aligned_head.astype(np.float32) * (1.0 - alpha), 0, 255
    ).astype(np.uint8)


def process_swap(input_path, template_path, output_path):
    detector, segmenter, transformer, blender = (
        FaceDetector(),
        HeadSegmenter(),
        HeadTransformer(),
        HeadBlender(),
    )
    img_in, img_tmp = cv2.imread(input_path), cv2.imread(template_path)
    if img_in is None or img_tmp is None:
        return False
    lms_in, lms_tmp = detector.get_landmarks(img_in), detector.get_landmarks(img_tmp)
    if lms_in is None or lms_tmp is None:
        return False
    y_offset = 6
    lms_tmp = np.array([(pt[0], pt[1] + y_offset) for pt in lms_tmp])
    mask_in, mask_tmp = (
        segmenter.get_head_mask(img_in),
        segmenter.get_head_mask(img_tmp),
    )
    mask_tmp_inpaint = prep_target_inpainting_mask(mask_tmp, lms_tmp)
    headless = blender.inpaint_template(img_tmp, mask_tmp_inpaint)
    aln_head, aln_mask_raw = transformer.align_head(
        img_in, mask_in, lms_in, img_tmp, lms_tmp
    )
    aln_mask_clean = clean_aligned_mask(aln_mask_raw, lms_tmp)
    aln_head = remove_white_fringe(aln_head, aln_mask_clean)
    aln_head = match_skin_tone_smart(aln_head, img_tmp, lms_tmp)
    headless_with_shadow = add_drop_shadow(headless, aln_mask_clean, lms_tmp)
    mask_norm = (aln_mask_clean / 255.0).astype(np.float32)[..., np.newaxis]
    final = (aln_head.astype(np.float32) * mask_norm) + (
        headless_with_shadow.astype(np.float32) * (1.0 - mask_norm)
    )
    final = np.clip(final, 0, 255).astype(np.uint8)
    final = apply_final_skin_filter(final, img_tmp, lms_tmp)
    cv2.imwrite(output_path, final)
    return True


if __name__ == "__main__":
    process_swap("input.jpg", "template.jpg", "final_head_swap.jpg")
