# Project Technical Stack & Citations
**Project:** IPL Avatar Generation Pipeline (Head-Swap)

This document outlines the libraries, frameworks, and methodologies employed in the development of the Avatar Generation Pipeline.

## 1. Core Technologies & Libraries

### Computer Vision & AI
*   **MediaPipe (Google):** Used for high-fidelity 3D face mesh detection and selfie segmentation.
    *   *Citation:* Google LLC. (2023). MediaPipe Solutions. https://developers.google.com/mediapipe
*   **OpenCV (Open Source Computer Vision Library):** The primary engine for image processing, including Affine Transformations, Inpainting, and Lab-space color normalization.
    *   *Citation:* Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
*   **NumPy:** Used for high-performance matrix operations and coordinate transformations.
    *   *Citation:* Harris, C.R., et al. (2020). Array programming with NumPy. Nature.

### Backend Infrastructure
*   **FastAPI:** High-performance web framework for the REST API.
*   **Uvicorn:** ASGI server implementation for running the FastAPI application.
*   **Docker & Docker Compose:** Containerization for environment parity and reproducible deployments.

### Frontend
*   **React (TypeScript):** UI framework for the interactive dashboard.
*   **Vite:** Next-generation frontend tooling for optimized builds and development.

## 2. Methodology & VFX Algorithms

### A. Geometric Alignment (Head Mapping)
The pipeline uses **Affine Transformations** based on key landmarks (eyes and chin) to align the source head onto the target's skeletal structure.

### B. Smart Skin Tone Matching
To solve the "lighting mismatch" problem, we implemented a statistical color transfer in the **CIE Lab color space**. This aligns the mean of the source skin tones to match the specific stadium lighting of the IPL templates.

### C. Advanced Blending & Inpainting
A custom blending algorithm was developed:
*   **Template Inpainting:** Uses `cv2.inpaint` to remove the original player's head from the template, creating a clean canvas for the swap.
*   **Drop Shadows:** Dynamically generates shadows under the new jawline to enhance realism.
*   **Fringe Removal:** Uses inpainting to remove white artifacts along the boundary of the swapped head.

### D. Signal Processing (Final Skin Filter)
To ensure the swap doesn't look "pasted on," we apply a final color wash and Gaussian-blurred mask to smoothly transition the skin tones between the new face and the original neck.

## 3. Engineering Challenges Resolved
*   **Docker Dependency Management:** Resolved `OSError: libEGL.so.1` by configuring Linux-level system dependencies within the `python:3.11-slim` Docker environment.
*   **Container Build Optimization:** Implemented multi-layer Docker caching to reduce image rebuild times by isolating system-level installations from application-level changes.
*   **Port Orchestration:** Managed host-to-container port mapping (5173) to resolve local development environment conflicts.

---
*Prepared for Internship Review - April 2026*
