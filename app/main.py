import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.model import extract_features, get_model
from app.pca_viz import compute_pca_visualization

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # warm up: load DINOv2 at startup
    yield


app = FastAPI(title="DINOv2 PCA Visualizer", lifespan=lifespan)


def _image_to_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/visualize")
async def visualize(files: list[UploadFile] = File(...)):
    from PIL import Image

    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    for f in files:
        if not f.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"'{f.filename}' is not an image.",
            )

    images = []
    for f in files:
        try:
            data = await f.read()
            images.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"Could not read '{f.filename}'."
            )

    # Extract features for all images
    features = [extract_features(img) for img in images]
    patch_tokens_list = [ft[0] for ft in features]
    num_patches_h = features[0][1]

    # Joint PCA over all images
    vis_images = compute_pca_visualization(patch_tokens_list, num_patches_h)

    results = []
    for img, vis in zip(images, vis_images):
        original_resized = img.resize((vis.width, vis.height))
        results.append({
            "original": _image_to_base64(original_resized),
            "visualization": _image_to_base64(vis),
        })

    return JSONResponse({"results": results})
