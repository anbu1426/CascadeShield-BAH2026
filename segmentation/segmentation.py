import torch
import cv2 as cv
import numpy as np
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    checkpoint_path = hf_hub_download(
        repo_id="anbu1426/my-kaggle-checkpoints", 
        filename="checkpoint_e25.pth",
        token=os.environ.get("HF_TOKEN") 
    )

    m = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    print(f"Model loaded — epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
    return m

# Loads automatically when imported
model = _load_model()


def run_segmentation(image_path):
    """
    Input:  path to satellite image (JPG/PNG/TIF, any size, RGB or 4-band)
    Output: binary numpy array shape (H, W), values 0 or 1
            1 = road, 0 = background
            same spatial dimensions as input image
    """
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image at path: {image_path}")

    # Handle 4-band Sentinel-2
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])
    img_resized = cv.resize(img, (512, 512))

    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    mask = cv.resize(mask, original_size, interpolation=cv.INTER_NEAREST)

    return mask  # shape (H, W), values 0 or 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = run_segmentation(sys.argv[1])
        print(f"Output shape:  {result.shape}")
        print(f"Unique values: {np.unique(result)}")
        print(f"Road coverage: {result.mean()*100:.2f}%")
    else:
        print("Usage: python segmentation.py <image_path>")
