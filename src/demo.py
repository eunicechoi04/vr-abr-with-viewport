import cv2
import json
import numpy as np
import math
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_PATH = '/Users/eunicechoi04/Downloads/videoabr/data/videos/sJxiPiAaB4k.mkv'
OUTPUT_PATH = 'output_visualized.mp4'
JSON_PATH = '/Users/eunicechoi04/Downloads/videoabr/output/05_95_tile_rankings/sJxiPiAaB4k_rankings.json'
VIEWPORT_DATA_PATH = '/Users/eunicechoi04/Downloads/videoabr/360_Video_analysis/data/uid-3ba968b8-887c-460e-a5f2-86295957d731/test0/Paris-sJxiPiAaB4k/Paris-sJxiPiAaB4k_0_processed.csv' # Assuming a list of {lat, lon, timestamp}

# Tile Grid Configuration (from your JSON)
ROWS = 4
COLS = 6
TOTAL_TILES = 24

# Visualization Settings
BLUR_STRENGTH = (51, 51) # Kernel size for low-quality blur (must be odd numbers)
HIGHLIGHT_COLOR = (0, 255, 0) # Green box for active tiles
VIEWPORT_COLOR = (0, 0, 255)  # Red dot for user gaze
DOT_RADIUS = 10

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_viewport_data(csv_path, target_fps=30):
    """
    Load viewport data from CSV and resample to target FPS.
    
    Args:
        csv_path: Path to CSV file with columns: timestamp, latitude_deg, longitude_deg
        target_fps: Target frames per second (default: 30)
    
    Returns:
        Dictionary mapping frame_number to (lat, lon) in degrees
        Format: { 0: (lat0, lon0), 1: (lat1, lon1), ... }
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Check required columns exist
        required_cols = ['timestamp', 'latitude_deg', 'longitude_deg']
        if not all(col in df.columns for col in required_cols):
            print(f"WARNING: CSV missing required columns: {required_cols}")
            return {}
        
        # Get time range
        max_time = df['timestamp'].max()
        
        # Create target timestamps at target_fps intervals
        frame_interval = 1.0 / target_fps  # e.g., 1/30 = 0.0333s per frame
        target_timestamps = np.arange(0, max_time + frame_interval, frame_interval)
        
        # Interpolate latitude and longitude to target timestamps
        lat_interp = np.interp(target_timestamps, df['timestamp'], df['latitude_deg'])
        lon_interp = np.interp(target_timestamps, df['timestamp'], df['longitude_deg'])
        
        # Create dictionary: frame_number -> (lat, lon)
        viewport_dict = {}
        for frame_num, (lat, lon) in enumerate(zip(lat_interp, lon_interp)):
            viewport_dict[frame_num] = (lat, lon)
        
        print(f"Loaded viewport data: {len(viewport_dict)} frames at {target_fps} FPS")
        print(f"Time range: 0s to {max_time:.2f}s")
        
        return viewport_dict
        
    except FileNotFoundError:
        print(f"ERROR: Viewport CSV not found: {csv_path}")
        return {}
    except Exception as e:
        print(f"ERROR loading viewport data: {e}")
        return {}

def get_tile_rect(tile_index, img_w, img_h, rows, cols):
    """
    Returns (x, y, w, h) for a specific tile index.
    Assumes row-major ordering (0 is top-left, rows-1 is bottom-right).
    """
    tile_w = img_w // cols
    tile_h = img_h // rows
    
    # Calculate row and col for this index
    r = tile_index // cols
    c = tile_index % cols
    
    x = c * tile_w
    y = r * tile_h
    
    return x, y, tile_w, tile_h

def lat_lon_to_pixel(lat, lon, img_w, img_h):
    """
    Converts 360 sphere coordinates to Equirectangular pixel coordinates.
    Assumes:
    Lon: -180 to 180 (Yaw) -> Maps to X
    Lat: -90 to 90 (Pitch)   -> Maps to Y
    """
    # Normalize Longitude (-180 to 180) to (0 to 1)
    # If your data is 0-360, adjust accordingly.
    norm_x = (lon + 180) / 360.0
    
    # Normalize Latitude (-90 to 90) to (0 to 1)
    # Note: In images, Y=0 is the TOP. Lat=90 is usually the TOP.
    norm_y = (90 - lat) / 180.0 
    
    x = int(norm_x * img_w)
    y = int(norm_y * img_h)
    
    # Clamp to boundaries
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    
    return x, y

# ==========================================
# 3. MAIN PROCESSING
# ==========================================

def process_video():
    # 1. Load Data
    with open(JSON_PATH, 'r') as f:
        ranking_data = json.load(f)
        
    # Load viewport data from CSV (sampled at 30 FPS)
    viewport_data = load_viewport_data(VIEWPORT_DATA_PATH, target_fps=30)
    
    if not viewport_data:
        print("ERROR: No viewport data loaded. Exiting.")
        return
    
    max_viewport_frames = max(viewport_data.keys())
    print(f"Will process {max_viewport_frames + 1} frames (limited by viewport data)")

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30  # JSON and viewport data are at 30 FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling interval
    frame_interval = original_fps / target_fps  # e.g., 60/30 = 2.0 means read every 2nd frame
    
    print(f"\nVideo FPS: {original_fps} → Target FPS: {target_fps}")
    print(f"Frame sampling interval: {frame_interval:.2f}")

    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, target_fps, (width, height))

    print(f"Processing frames (limited to viewport data)...")

    json_frame_idx = 0  # Index into JSON/viewport data (30 FPS)
    video_frame_idx = 0  # Index of frames read from video
    next_frame_to_process = 0.0  # Which video frame corresponds to next JSON frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this video frame matches a JSON/viewport frame
        if video_frame_idx < int(next_frame_to_process):
            video_frame_idx += 1
            continue  # Skip this frame
        
        # Stop processing if we've run out of viewport data
        if json_frame_idx not in viewport_data:
            print(f"\nStopping at JSON frame {json_frame_idx} (video frame {video_frame_idx}) - no more viewport data")
            break
            
        frame_key = f"frame_{json_frame_idx}"
        
        # --- A. Create Base (Low Quality) Image ---
        # We blur the whole frame to represent "Low Quality"
        # Alternatively, you could downscale and upscale to create pixelation
        low_quality_frame = cv2.GaussianBlur(frame, BLUR_STRENGTH, 0)
        final_frame = low_quality_frame.copy()

        # --- B. Render High Quality Tiles ---
        if frame_key in ranking_data['rankings']:
            top_tiles = ranking_data['rankings'][frame_key]['top_tiles']
            
            for tile_idx in top_tiles:
                tx, ty, tw, th = get_tile_rect(tile_idx, width, height, ROWS, COLS)
                
                # 1. Copy the SHARP pixels from original frame to the blurred final_frame
                final_frame[ty:ty+th, tx:tx+tw] = frame[ty:ty+th, tx:tx+tw]
                
                # 2. Draw Green Highlight Border
                cv2.rectangle(final_frame, (tx, ty), (tx+tw, ty+th), HIGHLIGHT_COLOR, 3)
                
                # # Optional: Add Text ID
                # cv2.putText(final_frame, str(tile_idx), (tx+10, ty+30), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, HIGHLIGHT_COLOR, 2)

        # --- C. Draw Viewport Gaze (User Head Position) ---
        # Get lat/lon for this frame (we already checked it exists)
        lat, lon = viewport_data[json_frame_idx]
        px, py = lat_lon_to_pixel(lat, lon, width, height)
        
        # Draw outer circle (white) and inner dot (red) for visibility
        cv2.circle(final_frame, (px, py), DOT_RADIUS + 2, (255, 255, 255), -1)
        cv2.circle(final_frame, (px, py), DOT_RADIUS, VIEWPORT_COLOR, -1)

        # Write frame
        out.write(final_frame)
        
        # Move to next JSON frame
        json_frame_idx += 1
        video_frame_idx += 1
        next_frame_to_process += frame_interval
        
        if json_frame_idx % 100 == 0:
            print(f"Processed {json_frame_idx} JSON frames (video frame {video_frame_idx})")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! Processed {json_frame_idx} JSON frames (from {video_frame_idx} video frames)")
    print(f"📹 Output saved to: {OUTPUT_PATH}")
    print(f"   Duration: {json_frame_idx / target_fps:.2f} seconds")

if __name__ == "__main__":
    process_video()