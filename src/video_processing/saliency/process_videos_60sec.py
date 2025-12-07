import cv2
import numpy as np
import os
import sys
import glob
import torch

# Add saliency directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'saliency'))
from infer import MLNet, preprocess_image

# Model setup
MODEL_PATH = "/Users/eunicechoi04/Downloads/videoabr/src/video_processing/saliency/2025-12-06 13_46_08.165973_saliency.model"
SHAPE_R = 240
SHAPE_C = 320
SHAPE_R_GT = 30
SHAPE_C_GT = 40
PRIOR_SIZE = (int(SHAPE_R_GT / 10), int(SHAPE_C_GT / 10))

# Setup device and model
device = torch.device("cpu")
print(f"Loading model from {MODEL_PATH}", flush=True)
model = MLNet(PRIOR_SIZE).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully\n", flush=True)

def process_frame_to_saliency(frame, original_width, original_height):
    """Process a single frame and return full-size saliency map"""
    # Resize frame to 240x320
    resized_frame = cv2.resize(frame, (SHAPE_C, SHAPE_R))

    # Save temporarily for preprocessing
    temp_path = "/tmp/temp_frame.jpg"
    cv2.imwrite(temp_path, resized_frame)

    # Preprocess using infer.py function
    img_processed, _ = preprocess_image(temp_path, SHAPE_R, SHAPE_C)

    # Convert to tensor
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
    output_map = pred.squeeze().cpu().numpy()
    output_map = (output_map - output_map.min()) / (output_map.max() - output_map.min() + 1e-8)
    output_map = (output_map * 255).astype(np.uint8)

    # Resize back to original dimensions
    output_map_fullsize = cv2.resize(output_map, (original_width, original_height))

    return output_map_fullsize

def process_video(video_path, output_dir, target_fps=30, max_duration=60):
    """Process first 60 seconds of video at 30fps and create saliency video"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_saliency_60s.mp4")

    print(f"Processing: {video_name}", flush=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}", flush=True)
        return

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Original dimensions: {original_width}x{original_height}", flush=True)
    print(f"  Original FPS: {original_fps}", flush=True)
    print(f"  Total frames in video: {total_frames}", flush=True)

    # Calculate max frames to process (60 seconds at target fps)
    max_frames = target_fps * max_duration
    print(f"  Will process: {max_frames} frames at {target_fps} fps", flush=True)

    # Calculate frame interval for sampling
    frame_interval = original_fps / target_fps

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (original_width, original_height), isColor=False)

    # Process frames
    frame_count = 0
    processed_count = 0
    next_frame_to_process = 0

    while processed_count < max_frames:
        # Set to the next frame we want to process
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(next_frame_to_process))
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame
        saliency_map = process_frame_to_saliency(frame, original_width, original_height)

        # Write to output video
        out.write(saliency_map)

        processed_count += 1
        next_frame_to_process += frame_interval

        if processed_count % 30 == 0:
            print(f"  Processed {processed_count}/{max_frames} frames ({100*processed_count/max_frames:.1f}%)", flush=True)

    # Cleanup
    cap.release()
    out.release()

    print(f"  Completed! Saved to: {output_path}", flush=True)
    print(f"  Final frame count: {processed_count}\n", flush=True)

def main():
    # Find all video files
    video_dir = "/Users/eunicechoi04/Downloads/videoabr/data/videos"
    output_dir = "/Users/eunicechoi04/Downloads/videoabr/data/saliency_videos_60s"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all .webm and .mkv files
    webm_files = glob.glob(os.path.join(video_dir, "*.webm"))
    mkv_files = glob.glob(os.path.join(video_dir, "*.mkv"))
    all_videos = webm_files + mkv_files

    print(f"Found {len(all_videos)} videos to process", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Processing: First 60 seconds at 30fps\n", flush=True)
    print("="*60, flush=True)

    # Process each video
    for i, video_path in enumerate(all_videos, 1):
        print(f"\n[{i}/{len(all_videos)}]", flush=True)
        process_video(video_path, output_dir, target_fps=30, max_duration=60)
        print("="*60, flush=True)

    print("\nAll videos processed successfully!", flush=True)

if __name__ == "__main__":
    main()
