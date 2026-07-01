
import torch
import cv2 as cv
import numpy as np
import segmentation_models_pytorch as smp

# Global model and device — loaded once when this module is imported
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # weights loaded from checkpoint, not imagenet
    in_channels=3,
    classes=1,
).to(device)

# Load your best checkpoint — update path as needed
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def run_segmentation(image_path):
    """
    Input:  path to any satellite image (JPG/PNG/TIF, any size, RGB or 4-band)
    Output: binary numpy array of shape (H, W), values 0 or 1
            where 1 = road, 0 = background
    """
    # Load image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image at path: {image_path}")

    # Handle 4-band Sentinel-2 images — drop NIR, keep RGB
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Convert BGR to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Store original size for restoring output dimensions
    original_size = (img.shape[1], img.shape[0])  # (W, H)

    # Resize to model input size
    img_resized = cv.resize(img, (512, 512))

    # Convert to tensor
    img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred = (pred > 0.5).float()

    # Convert to numpy, resize back to original dimensions
    mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    mask = cv.resize(mask, original_size, interpolation=cv.INTER_NEAREST)

    return mask  # shape (H, W), values 0 or 1


if __name__ == "__main__":
    # Quick test — replace with any image path you have
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        result = run_segmentation(test_path)
        print(f"Output shape: {result.shape}")
        print(f"Unique values: {np.unique(result)}")
        print(f"Road coverage: {result.mean()*100:.2f}%")
    else:
        print("Usage: python segmentation.py <image_path>")
