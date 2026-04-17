"""
Save PCA visualization results as a tiled image for README.

Usage:
    python save_results.py img1.jpg img2.jpg img3.jpg ...

Output:
    assets/result.png  -- tiled image: left=original, right=PCA visualization
                          one row per input image
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

from app.model import extract_features
from app.pca_viz import compute_pca_visualization

ASSETS_DIR = Path("assets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Input image paths")
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images]
    for p in image_paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    ASSETS_DIR.mkdir(exist_ok=True)

    print(f"Loading images ({len(image_paths)} files)...")
    images = [Image.open(p).convert("RGB") for p in image_paths]

    print("Extracting DINOv2 features...")
    features = [extract_features(img) for img in images]
    patch_tokens_list = [ft[0] for ft in features]
    num_patches_h = features[0][1]

    print("Computing PCA visualization...")
    vis_images = compute_pca_visualization(patch_tokens_list, num_patches_h)

    tile_w = vis_images[0].width
    tile_h = vis_images[0].height
    n = len(images)

    canvas = Image.new("RGB", (tile_w * 2, tile_h * n))
    for i, (img, vis) in enumerate(zip(images, vis_images)):
        canvas.paste(img.resize((tile_w, tile_h)), (0, i * tile_h))
        canvas.paste(vis, (tile_w, i * tile_h))

    out_path = ASSETS_DIR / "result.png"
    canvas.save(out_path)
    print(f"Saved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
