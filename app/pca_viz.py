import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

FOREGROUND_THRESHOLD = 0.6


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_pca_visualization(
    patch_tokens: torch.Tensor,
    num_patches_h: int,
    output_size: int = 448,
) -> Image.Image:
    """
    Two-stage PCA visualization:
      Stage 1: PCA(1 component) over all patches → foreground mask (threshold=0.6)
      Stage 2: PCA(3 components) over foreground patches only → RGB image

    Args:
        patch_tokens: (num_patches, embed_dim) tensor
        num_patches_h: number of patches per side (image is square)
        output_size: output image resolution in pixels

    Returns:
        PIL Image of the PCA visualization
    """
    tokens_np = patch_tokens.cpu().numpy()  # (N, D)
    N = tokens_np.shape[0]

    # --- Stage 1: background removal ---
    pca1 = PCA(n_components=1)
    first_component = pca1.fit_transform(tokens_np)[:, 0]  # (N,)
    first_component_norm = _minmax_normalize(first_component)
    foreground_mask = first_component_norm > FOREGROUND_THRESHOLD  # (N,) bool

    # PCA component direction is arbitrary — if the majority of patches are
    # marked as foreground, the sign is flipped (background > object area).
    if foreground_mask.sum() > N * 0.5:
        foreground_mask = ~foreground_mask

    # --- Stage 2: RGB visualization on foreground patches ---
    fg_tokens = tokens_np[foreground_mask]  # (M, D)

    rgb_patches = np.zeros((N, 3), dtype=np.float32)

    if fg_tokens.shape[0] >= 3:
        pca3 = PCA(n_components=3)
        fg_components = pca3.fit_transform(fg_tokens)  # (M, 3)
        for ch in range(3):
            fg_components[:, ch] = _minmax_normalize(fg_components[:, ch])
        rgb_patches[foreground_mask] = fg_components

    # --- Reshape to spatial grid ---
    vis_grid = rgb_patches.reshape(num_patches_h, num_patches_h, 3)  # (H_p, W_p, 3)
    vis_uint8 = (vis_grid * 255).astype(np.uint8)

    vis_image = Image.fromarray(vis_uint8, mode="RGB")
    vis_image = vis_image.resize((output_size, output_size), Image.NEAREST)
    return vis_image
