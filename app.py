import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from main import process_swap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")


@app.get("/api/templates")
async def get_templates():
    templates = [
        f
        for f in os.listdir("templates")
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return {"templates": templates}


@app.post("/api/swap")
async def swap_heads(
    input_image: UploadFile = File(...), template_name: str = Form(...)
):
    input_ext = os.path.splitext(input_image.filename)[1]
    input_filename = f"{uuid.uuid4()}{input_ext}"
    input_path = os.path.join("uploads", input_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(input_image.file, buffer)

    template_path = os.path.join("templates", template_name)
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template not found")

    output_filename = f"swap_{uuid.uuid4()}.jpg"
    output_path = os.path.join("outputs", output_filename)

    success = process_swap(input_path, template_path, output_path)

    if not success:
        raise HTTPException(
            status_code=500, detail="Head swap failed. Make sure a face is visible."
        )

    return {"output_url": f"/outputs/{output_filename}"}


@app.post("/api/upload-template")
async def upload_custom_template(template_image: UploadFile = File(...)):
    filename = f"custom_{uuid.uuid4()}_{template_image.filename}"
    path = os.path.join("templates", filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(template_image.file, buffer)
    return {"template_name": filename}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
