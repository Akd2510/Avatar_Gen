import cv2
import numpy as np
from detector import FaceDetector
from masking import MaskGenerator
from transform import FaceTransformer
from blender import ImageBlender

def match_skin_tone(source_face, target_img, mask):
    """
    Statistically aligns the face color to the target player's skin.
    Works entirely in OpenCV to avoid 'skimage' dependency errors.
    """
    # 1. Convert to Lab (Luminance, Green-Red, Blue-Yellow)
    src_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2Lab).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2Lab).astype(np.float32)

    # 2. Get masks for calculations
    s_idx = mask > 0
    if not np.any(s_idx): return source_face

    # 3. Calculate Mean and StdDev for each channel
    def get_stats(lab_img, idx):
        l, a, b = cv2.split(lab_img)
        return (np.mean(l[idx]), np.mean(a[idx]), np.mean(b[idx]),
                np.std(l[idx]), np.std(a[idx]), np.std(b[idx]))

    s_l_m, s_a_m, s_b_m, s_l_s, s_a_s, s_b_s = get_stats(src_lab, s_idx)
    # For target, we sample a broad area around the face
    t_l_m, t_a_m, t_b_m, t_l_s, t_a_s, t_b_s = get_stats(tgt_lab, s_idx)

    # 4. Reinhard Transfer Formula
    l, a, b = cv2.split(src_lab)
    l = ((l - s_l_m) * (t_l_s / (s_l_s + 1e-5))) + t_l_m
    a = ((a - s_a_m) * (t_a_s / (s_a_s + 1e-5))) + t_a_m
    b = ((b - s_b_m) * (t_b_s / (s_b_s + 1e-5))) + t_b_m

    # 5. Clip and Reconstruct
    res_lab = cv2.merge([np.clip(l,0,255), np.clip(a,0,255), np.clip(b,0,255)])
    res_bgr = cv2.cvtColor(res_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    final_face = np.zeros_like(source_face)
    final_face[s_idx] = res_bgr[s_idx]
    return final_face

def apply_pre_filter(image):
    """Normalizes the contrast and lighting distribution of an image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the Luminance channel to equalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    res_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(res_lab, cv2.COLOR_Lab2BGR)

def apply_glue_filter(image):
    """The 'Final Polish' that glues the face to the template."""
    # 1. Subtle Warming (Stadium Glow)
    # We slightly boost Red and decrease Blue globally
    img_float = image.astype(np.float32)
    img_float[:, :, 2] *= 1.02  # Red
    img_float[:, :, 0] *= 0.98  # Blue
    
    # 2. Add Digital Grain (The Secret Ingredient)
    # This hides the difference in pixel quality between the face and template
    noise = np.random.normal(0, 2, image.shape).astype(np.float32)
    img_float = cv2.add(img_float, noise)
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

def main():
    print("--- PIPELINE STARTING ---")
    
    # Loaders
    detector = FaceDetector()
    mask_gen = MaskGenerator() 
    transformer = FaceTransformer()
    blender = ImageBlender() 

    input_img = apply_pre_filter(cv2.imread('input.jpg')) # Filter at start
    template_img = apply_pre_filter(cv2.imread('template.jpg')) # Filter at start

    if input_img is None or template_img is None:
        print("ERROR: Could not find 'input.jpg' or 'template.jpg' in this folder.")
        return

    print("Step 1: Detecting Landmarks...")
    input_landmarks = detector.get_landmarks(input_img)
    template_landmarks = detector.get_landmarks(template_img)

    if input_landmarks is None or template_landmarks is None:
        print("ERROR: Face not detected or Gate triggered.")
        return

    print("Step 2: Generating Face Mask...")
    input_mask = mask_gen.create_mask(input_img.shape, input_landmarks)

    print("Step 3: Warping Face...")
    # NOTE: Check your transform.py to ensure it returns (face, mask)
    warped_face, warped_mask = transformer.warp_face(input_img, input_mask, input_landmarks, template_img, template_landmarks)

    # DEBUG: Check if we are swapping a face or a white blob
    print(f"DEBUG: Warped Image Shape: {warped_face.shape}")
    cv2.imwrite('debug_current_warped.jpg', warped_face)

    print("Step 4: Matching Skin Tone...")
    color_matched_face = match_skin_tone(warped_face, template_img, warped_mask)

    print("Step 5: Blending...")
    final_avatar = blender.blend(color_matched_face, template_img, warped_mask)

    final_avatar = apply_glue_filter(final_avatar) # Filter at end
    cv2.imwrite('final_ipl_avatar.jpg', final_avatar)
    print("--- SUCCESS: Image saved as final_ipl_avatar.jpg ---")

if __name__ == "__main__":
    main()