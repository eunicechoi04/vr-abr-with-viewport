import cv2
import numpy as np
import pandas as pd
import os
import glob

# CONFIGURATION
TILE_ROWS = 4
TILE_COLS = 6
FPS = 30  # Saliency videos are 30fps

def process_video_to_csv(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tile_h = height // TILE_ROWS
    tile_w = width // TILE_COLS
    
    results = []
    
    print(f"Processing {os.path.basename(video_path)} ({total_frames} frames)...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale if not already (Saliency maps are usually grayscale)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Calculate average brightness per tile
        # Normalize to 0.0 - 1.0
        frame_data = {'timestamp': frame_idx / FPS}
        
        for r in range(TILE_ROWS):
            for c in range(TILE_COLS):
                # Extract tile ROI (Region of Interest)
                y1, y2 = r * tile_h, (r + 1) * tile_h
                x1, x2 = c * tile_w, (c + 1) * tile_w
                tile_img = gray[y1:y2, x1:x2]
                
                # Average pixel intensity (0-255) -> (0.0-1.0)
                avg_val = np.mean(tile_img) / 255.0
                
                tile_id = r * TILE_COLS + c
                frame_data[f'tile_{tile_id}'] = avg_val
                
        results.append(frame_data)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...", end='\r')
            
    cap.release()
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved tile weights to {output_csv}")

if __name__ == "__main__":
    # Update to where your partner saved the .mp4 files
    SALIENCY_DIR = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\output\saliency_videos_60s"
    
    mp4_files = glob.glob(os.path.join(SALIENCY_DIR, "*.mp4"))
    for mp4 in mp4_files:
        out_name = mp4.replace(".mp4", "_weights.csv")
        process_video_to_csv(mp4, out_name)