import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import os

# ==========================================
# 1. Model Definition (Must match training)
# ==========================================
class MLNet(nn.Module):
    def __init__(self, prior_size):
        super(MLNet, self).__init__()
        # Load vgg16 structure. 
        # Note: We don't need pretrained weights here since we load your checkpoint later, 
        # but using pretrained=True ensures the structure matches exactly if that's what was used.
        features = list(models.vgg16(pretrained=False).features)[:-1]

        # Match specific stride/kernel/padding from the notebook
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2

        self.features = nn.ModuleList(features).eval()
        self.fddropout = nn.Dropout2d(p=0.5)
        self.int_conv = nn.Conv2d(1280, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.prior = nn.Parameter(torch.ones((1, 1, prior_size[0], prior_size[1]), requires_grad=True))
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=10)

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {16, 23, 29}:
                results.append(x)

        # Concat to get 1280 channels
        x = torch.cat((results[0], results[1], results[2]), 1)
        x = self.fddropout(x)
        x = self.int_conv(x)
        x = self.pre_final_conv(x)

        upscaled_prior = self.bilinearup(self.prior)
        
        # Dot product with prior
        x = x * upscaled_prior
        x = torch.nn.functional.relu(x, inplace=True)
        return x

# ==========================================
# 2. Preprocessing Functions
# ==========================================
def padding(img, shape_r, shape_c, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_image(img_path, shape_r, shape_c):
    original_image = cv2.imread(img_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image from {img_path}")
        
    padded_image = padding(original_image, shape_r, shape_c, 3)
    img = padded_image.astype('float')
    
    # cv2 is BGR, convert to RGB for model if needed, though original notebook comment says:
    # "# cv2 : BGR # PIL : RGB" then "ims = ims[...,::-1]" 
    # This line converts BGR (cv2 default) to RGB
    img = img[..., ::-1].copy() 
    
    img /= 255.0
    # (H, W, C) -> (C, H, W)
    img = np.rollaxis(img, 2, 0) 
    return img, original_image.shape

# ==========================================
# 3. Main Execution
# ==========================================
def run_inference(model_path, image_path, output_path, use_gpu=False):
    # These dimensions must match what you used during training.
    # Based on the notebook provided, the active config was:
    SHAPE_R = 240
    SHAPE_C = 320
    SHAPE_R_GT = 30
    SHAPE_C_GT = 40
    
    PRIOR_SIZE = (int(SHAPE_R_GT / 10), int(SHAPE_C_GT / 10))
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = MLNet(PRIOR_SIZE).to(device)
    
    # 2. Load Weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    # 3. Prepare Image
    print(f"Processing image: {image_path}")
    try:
        img_processed, original_shape = preprocess_image(image_path, SHAPE_R, SHAPE_C)
    except Exception as e:
        print(e)
        return

    # Convert to tensor and normalize
    img_tensor = torch.tensor(img_processed, dtype=torch.float).unsqueeze(0) # Add batch dim
    
    # Normalize using ImageNet stats (from notebook)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Squeeze, normalize, unsqueeze is needed because normalize expects 3 channels, not batch
    img_tensor[0] = normalize(img_tensor[0])
    img_tensor = img_tensor.to(device)

    # 4. Run Prediction
    with torch.no_grad():
        pred = model(img_tensor)

    # 5. Post-process and Save
    # Squeeze batch and channel dims: (1, 1, H, W) -> (H, W)
    output_map = pred.squeeze().cpu().numpy()
    
    # Normalize output to 0-255 for saving as image
    output_map = (output_map - output_map.min()) / (output_map.max() - output_map.min() + 1e-8)
    output_map = (output_map * 255).astype(np.uint8)
    
    # Optional: Resize back to original image size if desired
    # output_map = cv2.resize(output_map, (original_shape[1], original_shape[0]))

    cv2.imwrite(output_path, output_map)
    print(f"Saliency map saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLNet Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the .model file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save output saliency map")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    run_inference(args.model, args.image, args.output, args.gpu)