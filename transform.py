import cv2
import numpy as np

class FaceTransformer:
    def __init__(self):
        # We use the outer boundary indices to ensure the Delaunay triangulation
        # covers the entire face area, giving our tight inner-mask plenty of room to fit perfectly.
        self.boundary_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

    def _apply_affine_transform(self, src, src_tri, dst_tri, size):
        """Helper to warp a single triangle. Works dynamically for both images and masks."""
        r1 = cv2.boundingRect(np.float32([src_tri]))
        r2 = cv2.boundingRect(np.float32([dst_tri]))

        tri1_cropped = [((src_tri[i][0] - r1[0]), (src_tri[i][1] - r1[1])) for i in range(3)]
        tri2_cropped = [((dst_tri[i][0] - r2[0]), (dst_tri[i][1] - r2[1])) for i in range(3)]

        img1_cropped = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        warp_mat = cv2.getAffineTransform(np.float32(tri1_cropped), np.float32(tri2_cropped))
        
        img2_cropped = cv2.warpAffine(img1_cropped, warp_mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Create a generic polygon mask for the triangle
        mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2_cropped), 1.0, 16, 0)
        
        # Handle 3-channel (BGR) images vs 1-channel (Grayscale) masks seamlessly
        if len(src.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
        img2_cropped = img2_cropped * mask
        return img2_cropped, r2, mask

    def get_delaunay_triangles(self, rect, landmarks):
        """Generates triangulation indices based on landmarks."""
        subdiv = cv2.Subdiv2D(rect)
        for p in landmarks:
            # Explicitly cast to native Python integers to prevent OpenCV crash
            subdiv.insert((int(p[0]), int(p[1]))) 
        
        triangles_list = subdiv.getTriangleList()
        delaunay_tri = []
        
        for t in triangles_list:
            pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            pt_idx = []
            for i in range(3):
                for j in range(len(landmarks)):
                    if abs(pt[i][0] - landmarks[j][0]) < 1 and abs(pt[i][1] - landmarks[j][1]) < 1:
                        pt_idx.append(j)
                        break
            if len(pt_idx) == 3:
                delaunay_tri.append(pt_idx)
        return delaunay_tri

    def warp_face(self, input_image, input_mask, input_landmarks, template_image, template_landmarks):
        """Applies Delaunay warping to align both the input face and its soft mask."""
        h, w, channels = template_image.shape
        img_warped = np.zeros(template_image.shape, dtype=template_image.dtype)
        
        # Ensure the mask is processed as a 2D grayscale array
        if len(input_mask.shape) == 3:
            input_mask = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
        mask_warped = np.zeros((h, w), dtype=np.float32)

        src_boundary_pts = np.array([input_landmarks[idx] for idx in self.boundary_indices])
        dst_boundary_pts = np.array([template_landmarks[idx] for idx in self.boundary_indices])

        rect = (0, 0, w, h)
        dt = self.get_delaunay_triangles(rect, dst_boundary_pts)

        for tri_indices in dt:
            t1 = [src_boundary_pts[i] for i in tri_indices]
            t2 = [dst_boundary_pts[i] for i in tri_indices]

            # 1. Warp the image triangle
            img_triangle, r2, tri_mask_img = self._apply_affine_transform(input_image, t1, t2, (w, h))
            img_warped[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img_warped[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - tri_mask_img) + img_triangle

            # 2. Warp the soft mask triangle exactly the same way to preserve feathering
            mask_triangle, _, tri_mask_gray = self._apply_affine_transform(input_mask, t1, t2, (w, h))
            mask_warped[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = mask_warped[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - tri_mask_gray) + mask_triangle

        # We return only the warped face and the correctly warped soft mask
        return img_warped, mask_warped.astype(np.uint8)