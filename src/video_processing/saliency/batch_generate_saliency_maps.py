"""
Batch generate saliency map videos for all videos in the dataset.
This creates video files that can be used by the content-aware ABR simulator.
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# Import the model from infer.py (assuming it's in the same directory)
sys.path.append(os.path.dirname(__file__))
from infer import MLNet, preprocess_image


def generate_saliency_video(input_video_path, output_video_path, model, device):
    """
    Generate saliency map video from input video.

    Args:
        input_video_path: Path to input video
        output_video_path: Path to save saliency map video
        model: Trained MLNet model
        device: torch device (cuda or cpu)
    """
    print(f"Processing: {os.path.basename(input_video_path)}")

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"  ERROR: Could not open video")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Target: 30 FPS for 60 seconds
    TARGET_FPS = 30
    MAX_FRAMES = 60 * TARGET_FPS

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, TARGET_FPS, (width, height), isColor=False)

    frame_count = 0
    processed_count = 0

    # Model input dimensions (from infer.py)
    SHAPE_R = 240
    SHAPE_C = 320

    while frame_count < min(MAX_FRAMES, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily to process with model
        temp_frame_path = "/tmp/temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)

        try:
            # Preprocess and run model
            img_processed, original_shape = preprocess_image(temp_frame_path, SHAPE_R, SHAPE_C)
            img_tensor = torch.tensor(img_processed, dtype=torch.float).unsqueeze(0)

            # Normalize
            from torchvision import transforms
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            img_tensor[0] = normalize(img_tensor[0])
            img_tensor = img_tensor.to(device)

            # Run inference
            with torch.no_grad():
                pred = model(img_tensor)

            # Post-process
            saliency_map = pred.squeeze().cpu().numpy()
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            saliency_map = (saliency_map * 255).astype(np.uint8)

            # Resize to match original video dimensions
            saliency_map = cv2.resize(saliency_map, (width, height))

            # Write to output video
            out.write(saliency_map)
            processed_count += 1

        except Exception as e:
            print(f"  Error processing frame {frame_count}: {e}")
            # Write black frame on error
            out.write(np.zeros((height, width), dtype=np.uint8))

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{min(MAX_FRAMES, total_frames)} frames...")

    cap.release()
    out.release()

    # Clean up temp file
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    print(f"  Completed: {processed_count} frames")
    return True


def batch_generate(input_dir, output_dir, model_path, use_gpu=False):
    """
    Generate saliency maps for all videos in input directory.

    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save saliency map videos
        model_path: Path to trained model file
        use_gpu: Whether to use GPU
    """
    # Setup device and model
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    SHAPE_R_GT = 30
    SHAPE_C_GT = 40
    PRIOR_SIZE = (int(SHAPE_R_GT / 10), int(SHAPE_C_GT / 10))

    model = MLNet(PRIOR_SIZE).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from: {model_path}\n")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all video files
    input_path = Path(input_dir)
    video_files = list(input_path.glob("*.mkv")) + list(input_path.glob("*.webm")) + list(input_path.glob("*.mp4"))

    print(f"Found {len(video_files)} video files\n")
    print("=" * 60)

    # Process each video
    for i, video_path in enumerate(video_files, 1):
        video_id = video_path.stem
        output_filename = f"saliency_{video_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n[{i}/{len(video_files)}] {video_id}")

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"  SKIP: Output already exists")
            continue

        success = generate_saliency_video(
            str(video_path),
            output_path,
            model,
            device
        )

        if success:
            print(f"  Saved to: {output_filename}")
        else:
            print(f"  FAILED")

    print("\n" + "=" * 60)
    print("Batch processing complete!")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/data/videos"
    OUTPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/saliency_maps"
    MODEL_PATH = "/Users/eunicechoi04/Downloads/videoabr/src/video_processing/saliency/2025-12-06 13_46_08.165973_saliency.model"
    USE_GPU = False  # Set to True if you have CUDA available

    batch_generate(INPUT_DIR, OUTPUT_DIR, MODEL_PATH, USE_GPU)
