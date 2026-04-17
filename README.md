# Avatar Generator: Professional VFX Studio

A high-performance, anatomical-precision head swap pipeline built with Python and React. This project enables seamless face replacement while preserving real human features like neck creases, Adam's apples, and full beard volume.

![UI Preview](https://img.shields.io/badge/UI-Minimalist%20Studio-black)
![Backend](https://img.shields.io/badge/Backend-FastAPI-blue)
![VFX](https://img.shields.io/badge/VFX-OpenCV%20%2B%20MediaPipe-green)

## 🚀 Key Features

*   **Anatomical Preservation:** Unlike standard face-swappers, this pipeline identifies the template's neck anatomy and "melts" the new head into it, preserving realistic human skin creases.
*   **Universal Beard Shield:** Advanced masking ensures that wide beards and sideburns are never cut off, regardless of the target's jaw size.
*   **Shoulder-Kill Logic:** Surgically prunes source-image garments and backgrounds, ensuring zero "ghosting" from the original shirt.
*   **Lab-Space Color Matching:** Uses luminance-preserving color shifts to match skin tones perfectly while keeping 100% of the original skin texture and pores.
*   **Custom Template Support:** Upload your own body templates directly through the UI to expand your gallery instantly.
*   **Modern Studio UI:** A minimalist, professional interface designed for a streamlined VFX workflow.

## 🛠️ Tech Stack

*   **Backend:** Python 3.11, FastAPI, OpenCV, MediaPipe (Anatomy Tracking).
*   **Frontend:** React (TypeScript), Vite, Vanilla CSS (Studio Aesthetic), Axios.
*   **ML Models:** MediaPipe Face Landmarker & Selfie Segmenter.

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone git@github.com:Akd2510/Avatar_Gen.git
cd Avatar_Gen
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API
python app.py
```
*Note: Ensure `face_landmarker.task` and `selfie_segmenter.tflite` are present in the root directory.*

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The UI will be available at `http://localhost:5173`.

## 🧠 How it Works

The pipeline follows a sophisticated 5-stage VFX process:
1.  **Detection:** MediaPipe extracts 478 3D landmarks for both source and target.
2.  **Alignment:** A partial-affine transformation aligns the source head to the target's skeletal structure without stretching.
3.  **Surgical Masking:** A dual-zone mask applies wide feathering to the neck skin join and sharp pruning to the shoulder/garment areas.
4.  **Color Harmonization:** A Lab-space mean-shift aligns the source skin color to the template's specific lighting.
5.  **Grain Sync:** Subtle noise is injected into the swap to match the high-frequency film grain of the template image.

---
Developed for professional-grade avatar generation.
