"""Generate saliency heatmap for an image"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Add video_processing to path for importing MLNet
sys.path.insert(0, str(Path(__file__).parent / "video_processing" / "saliency"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
from mlnet import MLNet

def padding(img, shape_r, shape_c, channels=3):
    """Pad image to target shape while maintaining aspect ratio"""
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


def preprocess_image(img, shape_r, shape_c):
    """Preprocess image for MLNet model"""
    padded_image = padding(img, shape_r, shape_c, 3)
    img_processed = padded_image.astype('float')
    
    # Convert BGR to RGB
    img_processed = img_processed[..., ::-1].copy()
    img_processed /= 255.0
    
    # (H, W, C) -> (C, H, W)
    img_processed = np.rollaxis(img_processed, 2, 0)
    return img_processed


def generate_saliency_map(image_path, model_path, use_gpu=False):
    """
    Generate saliency map using MLNet model.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained MLNet model
        use_gpu: Whether to use GPU
    
    Returns:
        image: Original image (BGR)
        saliency_map: Grayscale saliency map (0-255)
    """
    # Model parameters (must match training)
    SHAPE_R = 240
    SHAPE_C = 320
    SHAPE_R_GT = 30
    SHAPE_C_GT = 40
    PRIOR_SIZE = (int(SHAPE_R_GT / 10), int(SHAPE_C_GT / 10))
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    original_shape = image.shape
    
    # Load model
    model = MLNet(PRIOR_SIZE).to(device)
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Preprocess image
    img_processed = preprocess_image(image, SHAPE_R, SHAPE_C)
    img_tensor = torch.tensor(img_processed, dtype=torch.float).unsqueeze(0)
    
    # Normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_tensor[0] = normalize(img_tensor[0])
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)
    
    # Post-process
    saliency_map = pred.squeeze().cpu().numpy()
    
    # Normalize to 0-255
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    saliency_map = (saliency_map * 255).astype(np.uint8)
    
    # Resize to original image size
    saliency_map = cv2.resize(saliency_map, (original_shape[1], original_shape[0]))
    
    return image, saliency_map


def apply_orange_heatmap(image, saliency_map, alpha=0.6):
    """
    Apply orange-colored heatmap overlay on image.
    
    Args:
        image: Original image (BGR)
        saliency_map: Grayscale saliency map (0-255)
        alpha: Overlay transparency (0=transparent, 1=opaque)
    
    Returns:
        overlayed_image: Image with orange heatmap overlay
    """
    # Resize saliency map to match image size if needed
    if saliency_map.shape[:2] != image.shape[:2]:
        saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    
    # Create custom orange colormap
    # Map: low values (0) -> dark/transparent, high values (255) -> bright orange
    orange_colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        # Orange gradient: (0, 69, 255) in BGR = (255, 165, 0) in RGB
        intensity = i / 255.0
        orange_colormap[i, 0, 0] = int(0 * intensity)      # Blue channel (low)
        orange_colormap[i, 0, 1] = int(165 * intensity)    # Green channel (medium)
        orange_colormap[i, 0, 2] = int(255 * intensity)    # Red channel (high)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(saliency_map, orange_colormap)
    
    # Blend with original image
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed, heatmap


def visualize_results(original, saliency_map, heatmap, overlayed, save_path=None):
    """
    Create visualization with 4 subplots showing the process.
    
    Args:
        original: Original image
        saliency_map: Grayscale saliency map
        heatmap: Orange heatmap
        overlayed: Final overlayed result
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
    
    # Plot original image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Plot grayscale saliency map
    axes[0, 1].imshow(saliency_map, cmap='gray')
    axes[0, 1].set_title('Saliency Map (Grayscale)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot orange heatmap
    axes[1, 0].imshow(heatmap_rgb)
    axes[1, 0].set_title('Orange Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot overlayed result
    axes[1, 1].imshow(overlayed_rgb)
    axes[1, 1].set_title('Overlayed Result', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """Main execution"""
    # Input image path
    image_path = Path("/Users/eunicechoi04/Downloads/6.5820/videoabr/output/sample/COCO_test2014_000000336914.jpg")
    
    # Model path
    model_path = Path("/Users/eunicechoi04/Downloads/6.5820/videoabr/src/video_processing/saliency/2025-12-06 13_46_08.165973_saliency.model")
    
    # Output directory
    output_dir = image_path.parent / "saliency_heatmaps"
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("SALIENCY HEATMAP GENERATOR (MLNet)")
    print("="*60)
    print(f"Input image: {image_path.name}")
    print(f"Model: {model_path.name}")
    print(f"Output directory: {output_dir}")
    print("-"*60)
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return
    
    # Step 1: Generate saliency map
    print("\n[1] Generating saliency map with MLNet...")
    try:
        original, saliency_map = generate_saliency_map(image_path, model_path, use_gpu=False)
        print(f"    ✓ Saliency map generated")
        print(f"    - Image size: {original.shape[1]}x{original.shape[0]}")
        print(f"    - Saliency range: [{saliency_map.min()}, {saliency_map.max()}]")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Apply orange heatmap
    print("\n[2] Applying orange heatmap overlay...")
    overlayed, heatmap = apply_orange_heatmap(original, saliency_map, alpha=0.6)
    print(f"    ✓ Heatmap applied with alpha=0.6")
    
    # Step 3: Save outputs
    print("\n[3] Saving outputs...")
    
    # Save individual outputs
    saliency_output = output_dir / f"{image_path.stem}_saliency.png"
    cv2.imwrite(str(saliency_output), saliency_map)
    print(f"    ✓ Saved saliency map: {saliency_output.name}")
    
    heatmap_output = output_dir / f"{image_path.stem}_heatmap.png"
    cv2.imwrite(str(heatmap_output), heatmap)
    print(f"    ✓ Saved orange heatmap: {heatmap_output.name}")
    
    overlayed_output = output_dir / f"{image_path.stem}_overlayed.png"
    cv2.imwrite(str(overlayed_output), overlayed)
    print(f"    ✓ Saved overlayed result: {overlayed_output.name}")
    
    # Step 4: Create visualization
    print("\n[4] Creating visualization...")
    viz_output = output_dir / f"{image_path.stem}_visualization.png"
    visualize_results(original, saliency_map, heatmap, overlayed, save_path=viz_output)
    
    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  - {saliency_output.name}")
    print(f"  - {heatmap_output.name}")
    print(f"  - {overlayed_output.name}")
    print(f"  - {viz_output.name}")


if __name__ == "__main__":
    main()
