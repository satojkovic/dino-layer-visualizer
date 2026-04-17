import torch
import torchvision.transforms as T
from PIL import Image

PATCH_SIZE = 14
IMAGE_SIZE = 448  # 14 * 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        _model.eval()
        _model.to(DEVICE)
    return _model


def preprocess(image: Image.Image) -> torch.Tensor:
    resize_size = IMAGE_SIZE + int(IMAGE_SIZE * 0.01) * 10  # 448 -> 488
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(resize_size),
        T.CenterCrop(IMAGE_SIZE),
        T.Normalize([0.5], [0.5]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)


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
