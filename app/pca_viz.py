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
    patch_tokens_list: list[torch.Tensor],
    num_patches_h: int,
    output_size: int = 448,
) -> list[Image.Image]:
    """
    Two-stage PCA visualization over multiple images jointly:
      Stage 1: PCA(1 component) over all patches from all images → foreground masks
      Stage 2: PCA(3 components) over all foreground patches jointly → RGB images

    Processing jointly across images ensures consistent color mapping and
    better foreground/background separation (same as DINOv2_pca_visualization/main.py).

    Args:
        patch_tokens_list: list of (num_patches, embed_dim) tensors, one per image
        num_patches_h: number of patches per side (image is square)
        output_size: output image resolution in pixels

    Returns:
        List of PIL Images of PCA visualizations, one per input image
    """
    tokens_list = [t.cpu().numpy() for t in patch_tokens_list]  # list of (N, D)
    N = tokens_list[0].shape[0]  # patches per image

    # --- Stage 1: foreground/background separation across all images ---
    all_tokens = np.vstack(tokens_list)  # (num_images * N, D)
    pca1 = PCA(n_components=1)
    first_component = pca1.fit_transform(all_tokens)[:, 0]  # (num_images * N,)
    first_component_norm = _minmax_normalize(first_component)

    # Determine mask direction using corner patches (corners are almost always background).
    # If the majority of corner patches score above the threshold, the PCA component
    # direction is inverted — use < threshold instead of > threshold.
    corner_indices = [
        0, num_patches_h - 1,
        N - num_patches_h, N - 1,
    ]
    corner_scores = np.concatenate([
        first_component_norm[i * N:(i + 1) * N][corner_indices]
        for i in range(len(tokens_list))
    ])
    inverted = (corner_scores > FOREGROUND_THRESHOLD).mean() > 0.5

    # Split back per image and compute masks
    masks = []
    for i in range(len(tokens_list)):
        patch_scores = first_component_norm[i * N:(i + 1) * N]
        mask = patch_scores > FOREGROUND_THRESHOLD
        if inverted:
            mask = ~mask
        masks.append(mask)

    # --- Stage 2: RGB visualization on foreground patches from all images jointly ---
    fg_tokens_per_image = [tokens_list[i][masks[i]] for i in range(len(tokens_list))]
    all_fg_tokens = np.vstack(fg_tokens_per_image)  # (total_fg_patches, D)

    images = []

    if all_fg_tokens.shape[0] >= 3:
        pca3 = PCA(n_components=3)
        all_fg_components = pca3.fit_transform(all_fg_tokens)  # (total_fg, 3)
        # Normalize each channel globally across all images
        for ch in range(3):
            all_fg_components[:, ch] = _minmax_normalize(all_fg_components[:, ch])

        # Split foreground components back per image
        split_indices = np.cumsum([fg.shape[0] for fg in fg_tokens_per_image[:-1]])
        fg_components_per_image = np.split(all_fg_components, split_indices)

        for i in range(len(tokens_list)):
            rgb_patches = np.zeros((N, 3), dtype=np.float32)
            rgb_patches[masks[i]] = fg_components_per_image[i]
            vis_grid = rgb_patches.reshape(num_patches_h, num_patches_h, 3)
            vis_uint8 = (vis_grid * 255).astype(np.uint8)
            vis_image = Image.fromarray(vis_uint8, mode="RGB")
            vis_image = vis_image.resize((output_size, output_size), Image.NEAREST)
            images.append(vis_image)
    else:
        # Fallback: return black images if not enough foreground patches
        for _ in range(len(tokens_list)):
            images.append(Image.fromarray(
                np.zeros((output_size, output_size, 3), dtype=np.uint8), mode="RGB"
            ))

    return images
