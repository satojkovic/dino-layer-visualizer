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
async def visualize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    from PIL import Image

    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    patch_tokens, num_patches_h = extract_features(image)
    vis_image = compute_pca_visualization(patch_tokens, num_patches_h)

    # Resize original to match visualization size for side-by-side display
    original_resized = image.resize((vis_image.width, vis_image.height))

    return JSONResponse({
        "original": _image_to_base64(original_resized),
        "visualization": _image_to_base64(vis_image),
    })
