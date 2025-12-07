import cv2
import numpy as np
import os

def create_motion_map(input_folder, output_folder, threshold=2.0):
    """
    Generates motion map videos for the first 60 seconds of all videos in input_folder.
    
    Args:
        input_folder (str): Path to source videos.
        output_folder (str): Path to save motion maps.
        threshold (float): Magnitude threshold for motion detection (removes noise).
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all video files (extensions can be added as needed)
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_filename = f"motion_{os.path.splitext(video_file)[0]}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Processing: {video_file}...")

        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error opening {video_file}")
            continue

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Methodology constraint: 30 FPS for 60 seconds = 1800 frames max
        TARGET_FPS = 30
        MAX_FRAMES = 60 * TARGET_FPS 
        
        # Define codec for output (MP4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height), isColor=False)

        # Read the first frame
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            continue

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0

        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- CORE METHODOLOGY: Optical Flow ---
            # Using Farneback's method to estimate dense optical flow
            # Parameters: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Calculate magnitude and angle of 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # --- BINARY MAPPING ---
            # Paper: "white pixel indicates the pixel is on one of the optical flows"
            # We apply a threshold to ignore sensor noise/compression artifacts
            _, motion_mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
            
            # Ensure it is uint8
            motion_mask = motion_mask.astype(np.uint8)

            # Write the binary frame to output video
            out.write(motion_mask)

            # Update previous frame
            prev_gray = gray
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"  > Processed {frame_count} frames...")

        # Release resources
        cap.release()
        out.release()
        print(f"Finished {video_file}. Saved to {output_path}")

if __name__ == "__main__":
    # CONFIGURATION
    INPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/data/videos"
    OUTPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/motion_maps"
    
    create_motion_map(INPUT_DIR, OUTPUT_DIR)