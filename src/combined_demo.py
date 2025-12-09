import cv2
import json
import numpy as np
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_PATH = '/Users/eunicechoi04/Downloads/videoabr/data/videos/sJxiPiAaB4k.mkv'
OUTPUT_PATH = 'combined_output_visualized.mp4'
JSON_PATH = '/Users/eunicechoi04/Downloads/videoabr/output/viewport_aware_tile_rankings/sJxiPiAaB4k_uid-3ba968b8-887c-460e-a5f2-86295957d731_test0_rankings.json'
ROWS = 4
COLS = 6
TOTAL_TILES = 24

# Visualization Settings
BLUR_STRENGTH = (51, 51)      # Kernel size for low-quality blur (must be odd numbers)
HIGH_QUALITY_COLOR = (0, 255, 0)  # Green box for high-quality tiles
LOW_QUALITY_COLOR = (128, 128, 128)  # Gray overlay for low-quality tiles
VIEWPORT_COLOR = (0, 0, 255)  # Red dot for user gaze
DOT_RADIUS = 12
BORDER_THICKNESS = 4

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

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
    # 1. Load tile rankings JSON
    print("Loading tile rankings...")
    with open(JSON_PATH, 'r') as f:
        ranking_data = json.load(f)
    
    print(f"Loaded rankings for {ranking_data['total_frames']} frames")
    print(f"Video: {ranking_data['video_id']}")
    print(f"User: {ranking_data['user_id']}")
    print(f"Top tiles per frame: {ranking_data['tile_config']['top_n_tiles']}")
    print(f"Weights - Motion: {ranking_data['weights']['motion']}, Saliency: {ranking_data['weights']['saliency']}, Viewport: {ranking_data['weights']['viewport']}")
    
    max_frames = ranking_data['total_frames']

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30  # JSON is sampled at 30 FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling interval
    frame_interval = original_fps / target_fps  # e.g., 60/30 = 2.0 means read every 2nd frame
    
    print(f"\nVideo FPS: {original_fps} → JSON FPS: {target_fps}")
    print(f"Frame sampling interval: {frame_interval:.2f}")

    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, target_fps, (width, height))

    print(f"\nProcessing up to {max_frames} frames (limited by ranking data)...")
    print(f"Video resolution: {width}x{height}")

    json_frame_idx = 0  # Index into JSON data (30 FPS)
    video_frame_idx = 0  # Index of frames read from video
    next_frame_to_process = 0.0  # Which video frame corresponds to next JSON frame
    
    while cap.isOpened() and json_frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this video frame matches a JSON frame
        if video_frame_idx < int(next_frame_to_process):
            video_frame_idx += 1
            continue  # Skip this frame
        
        frame_key = f"frame_{json_frame_idx}"
        
        # Check if we have ranking data for this frame
        if frame_key not in ranking_data['rankings']:
            print(f"\nNo ranking data for {frame_key}, stopping")
            break
        
        frame_data = ranking_data['rankings'][frame_key]
        top_tiles = set(frame_data['top_tiles'])
        
        # --- A. Start with blurred frame (low quality baseline) ---
        low_quality_frame = cv2.GaussianBlur(frame, BLUR_STRENGTH, 0)
        final_frame = low_quality_frame.copy()

        # --- B. Render tiles based on quality ---
        for tile_idx in range(TOTAL_TILES):
            tx, ty, tw, th = get_tile_rect(tile_idx, width, height, ROWS, COLS)
            
            if tile_idx in top_tiles:
                # HIGH QUALITY TILE: Use sharp original pixels
                final_frame[ty:ty+th, tx:tx+tw] = frame[ty:ty+th, tx:tx+tw]
                
                # Draw green border to indicate high quality
                cv2.rectangle(final_frame, (tx, ty), (tx+tw, ty+th), 
                            HIGH_QUALITY_COLOR, BORDER_THICKNESS)
            else:
                # LOW QUALITY TILE: Keep blurred, add subtle gray border
                cv2.rectangle(final_frame, (tx, ty), (tx+tw, ty+th), 
                            LOW_QUALITY_COLOR, 2)

        # --- C. Draw Viewport Position (from JSON data) ---
        if 'viewport_position' in frame_data:
            lat = frame_data['viewport_position']['latitude']
            lon = frame_data['viewport_position']['longitude']
            px, py = lat_lon_to_pixel(lat, lon, width, height)
            
            # Draw viewport indicator: white outer circle, red inner dot
            cv2.circle(final_frame, (px, py), DOT_RADIUS + 3, (255, 255, 255), -1)
            cv2.circle(final_frame, (px, py), DOT_RADIUS, VIEWPORT_COLOR, -1)
            
            # Optional: Add crosshair for precise location
            cv2.line(final_frame, (px - 15, py), (px + 15, py), (255, 255, 255), 2)
            cv2.line(final_frame, (px, py - 15), (px, py + 15), (255, 255, 255), 2)

        # --- D. Add info overlay ---
        info_text = f"JSON Frame: {json_frame_idx} | Video Frame: {video_frame_idx} | High-Quality: {len(top_tiles)}/{TOTAL_TILES}"
        cv2.putText(final_frame, info_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add viewport info if available
        if 'viewport_position' in frame_data:
            viewport_text = f"Viewport: ({lat:.1f}, {lon:.1f})"
            cv2.putText(final_frame, viewport_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Write frame
        out.write(final_frame)
        
        # Move to next JSON frame
        json_frame_idx += 1
        video_frame_idx += 1
        next_frame_to_process += frame_interval
        
        if json_frame_idx % 100 == 0:
            print(f"Processed {json_frame_idx}/{max_frames} JSON frames (video frame {video_frame_idx})")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Done! Processed {json_frame_idx} JSON frames (from {video_frame_idx} video frames)")
    print(f"📹 Output saved to: {OUTPUT_PATH}")
    print(f"   Duration: {json_frame_idx / target_fps:.2f} seconds")

if __name__ == "__main__":
    process_video()