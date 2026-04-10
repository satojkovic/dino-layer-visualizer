import torch
import torchvision.transforms as T
from PIL import Image

PATCH_SIZE = 14
IMAGE_SIZE = 448  # 14 * 32

_model = None


def get_model():
    global _model
    if _model is None:
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _model.eval()
    return _model


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def extract_features(image: Image.Image) -> tuple[torch.Tensor, int]:
    """
    Returns:
        patch_tokens: Tensor of shape (num_patches, embed_dim)
        num_patches_h: number of patches along height (= width)
    """
    model = get_model()
    tensor = preprocess(image)

    with torch.no_grad():
        features = model.forward_features(tensor)

    patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (num_patches, embed_dim)
    num_patches_h = IMAGE_SIZE // PATCH_SIZE
    return patch_tokens, num_patches_h
