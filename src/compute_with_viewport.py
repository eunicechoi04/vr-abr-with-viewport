import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# ==========================================
# CONFIGURATION
# ==========================================
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
FPS = 30
SEGMENT_DURATION = 1.0  # seconds

# Weighting factors for tile selection (ADJUST THESE)
MOTION_WEIGHT = 0.005
SALIENCY_WEIGHT = 0.095
VIEWPORT_WEIGHT = 0.9
TOP_N_TILES = 9  # Number of tiles to mark as high priority

# Paths
MOTION_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/motion_maps"
SALIENCY_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/saliency_videos_60s"
VIEWPORT_DIR = "/Users/eunicechoi04/Downloads/videoabr/360_Video_analysis/data"
OUTPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/viewport_aware_tile_rankings"


# ==========================================
# TILE ANALYSIS FUNCTIONS
# ==========================================
def get_tile_boundaries(frame_height, frame_width):
    """
    Calculate pixel boundaries for each tile in a 4x6 grid.
    Returns: List of (row_start, row_end, col_start, col_end) for each tile
    """
    tile_height = frame_height // TILE_ROWS
    tile_width = frame_width // TILE_COLS

    tiles = []
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            r_start = row * tile_height
            r_end = (row + 1) * tile_height
            c_start = col * tile_width
            c_end = (col + 1) * tile_width
            tiles.append((r_start, r_end, c_start, c_end))

    return tiles


def calculate_motion_scores(motion_frame, tile_boundaries):
    """
    Calculate motion intensity for each tile.
    Motion map has white pixels where motion is detected.
    """
    scores = []
    for (r_start, r_end, c_start, c_end) in tile_boundaries:
        tile_region = motion_frame[r_start:r_end, c_start:c_end]
        motion_score = np.mean(tile_region) / 255.0
        scores.append(motion_score)
    return np.array(scores)


def calculate_saliency_scores(saliency_frame, tile_boundaries):
    """
    Calculate average saliency for each tile.
    Saliency map is a heatmap where brighter = more salient.
    """
    scores = []
    for (r_start, r_end, c_start, c_end) in tile_boundaries:
        tile_region = saliency_frame[r_start:r_end, c_start:c_end]
        saliency_score = np.mean(tile_region) / 255.0
        scores.append(saliency_score)
    return np.array(scores)


def lat_lon_to_tile(lat, lon, frame_width, frame_height):
    """
    Convert latitude/longitude to tile index.
    
    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)
        frame_width: Video frame width
        frame_height: Video frame height
    
    Returns:
        Tile index (0-23 for 4x6 grid)
    """
    # Convert lat/lon to pixel coordinates
    norm_x = (lon + 180) / 360.0
    norm_y = (90 - lat) / 180.0
    
    x = int(norm_x * frame_width)
    y = int(norm_y * frame_height)
    
    # Clamp to boundaries
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    
    # Convert pixel to tile
    tile_width = frame_width // TILE_COLS
    tile_height = frame_height // TILE_ROWS
    
    tile_col = min(x // tile_width, TILE_COLS - 1)
    tile_row = min(y // tile_height, TILE_ROWS - 1)
    
    tile_index = tile_row * TILE_COLS + tile_col
    
    return tile_index


def get_neighboring_tiles(tile_index):
    """
    Get neighboring tiles (up, down, left, right, and diagonals).
    
    Args:
        tile_index: Index of the center tile (0-23)
    
    Returns:
        List of neighboring tile indices
    """
    row = tile_index // TILE_COLS
    col = tile_index % TILE_COLS
    
    neighbors = []
    
    # Check all 8 directions
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip the center tile itself
            
            new_row = row + dr
            new_col = col + dc
            
            # Check if within bounds
            if 0 <= new_row < TILE_ROWS and 0 <= new_col < TILE_COLS:
                neighbor_idx = new_row * TILE_COLS + new_col
                neighbors.append(neighbor_idx)
    
    return neighbors


def calculate_viewport_scores(lat, lon, frame_width, frame_height):
    """
    Calculate viewport-based scores for all tiles.
    
    Args:
        lat: Latitude of viewport center in degrees
        lon: Longitude of viewport center in degrees
        frame_width: Video frame width
        frame_height: Video frame height
    
    Returns:
        Array of viewport scores for each tile (0.0, 0.5, or 1.0)
    """
    scores = np.zeros(TOTAL_TILES)
    
    # Find the tile the user is looking at
    center_tile = lat_lon_to_tile(lat, lon, frame_width, frame_height)
    
    # Assign scores
    scores[center_tile] = 1.0  # Center tile gets full score
    
    # Neighboring tiles get 0.5
    neighbors = get_neighboring_tiles(center_tile)
    for neighbor_idx in neighbors:
        scores[neighbor_idx] = 0.5
    
    # All other tiles remain 0.0
    
    return scores


def load_viewport_data(csv_path, target_fps=30):
    """
    Load viewport data from CSV and resample to target FPS.
    
    Args:
        csv_path: Path to CSV file with columns: timestamp, latitude_deg, longitude_deg
        target_fps: Target frames per second (default: 30)
    
    Returns:
        Dictionary mapping frame_number to (lat, lon) in degrees
    """
    try:
        df = pd.read_csv(csv_path)
        
        required_cols = ['timestamp', 'latitude_deg', 'longitude_deg']
        if not all(col in df.columns for col in required_cols):
            print(f"  WARNING: CSV missing required columns")
            return {}
        
        max_time = df['timestamp'].max()
        
        # Resample to target FPS
        frame_interval = 1.0 / target_fps
        target_timestamps = np.arange(0, max_time + frame_interval, frame_interval)
        
        lat_interp = np.interp(target_timestamps, df['timestamp'], df['latitude_deg'])
        lon_interp = np.interp(target_timestamps, df['timestamp'], df['longitude_deg'])
        
        viewport_dict = {}
        for frame_num, (lat, lon) in enumerate(zip(lat_interp, lon_interp)):
            viewport_dict[frame_num] = (lat, lon)
        
        return viewport_dict
        
    except FileNotFoundError:
        print(f"  ERROR: Viewport CSV not found: {csv_path}")
        return {}
    except Exception as e:
        print(f"  ERROR loading viewport data: {e}")
        return {}


def select_tiles_viewport_aware(motion_scores, saliency_scores, viewport_scores, top_n=TOP_N_TILES):
    """
    Combine motion, saliency, and viewport scores to select top N tiles.
    
    Args:
        motion_scores: Array of motion scores for each tile
        saliency_scores: Array of saliency scores for each tile
        viewport_scores: Array of viewport scores for each tile (0.0, 0.5, or 1.0)
        top_n: Number of top tiles to select
    
    Returns:
        Tuple of (top_tile_indices, combined_scores)
    """
    if np.max(motion_scores) > 0:
        norm_motion = motion_scores / np.max(motion_scores)
    else:
        norm_motion = motion_scores # Handle case where frame is all black

    # 2. Normalize Saliency Scores to 0-1 range based on the frame's max
    if np.max(saliency_scores) > 0:
        norm_saliency = saliency_scores / np.max(saliency_scores)
    else:
        norm_saliency = saliency_scores

    # Viewport is already 0, 0.5, 1.0, so it doesn't need scaling
    norm_viewport = viewport_scores

    # 3. Combine scores with equal weights
    # Note: If you want exactly equal weights, use 1.0 for all (or 0.33)
    # Since we are ranking, the scale of the weights doesn't matter, only the ratio.
    combined_scores = (MOTION_WEIGHT * norm_motion +
                      SALIENCY_WEIGHT * norm_saliency +
                      VIEWPORT_WEIGHT * norm_viewport)
    
    # Get indices of top N tiles
    top_tile_indices = np.argsort(combined_scores)[-top_n:]
    
    # Return reversed list (highest score first) for better readability
    return top_tile_indices.tolist()[::-1], combined_scores


# ==========================================
# MAIN PRE-COMPUTATION FUNCTION
# ==========================================
def precompute_viewport_aware_rankings(motion_dir, saliency_dir, viewport_dir, output_dir, video_id, user_id, test_id):
    """
    Pre-compute viewport-aware tile rankings for a specific video and user.
    
    Args:
        motion_dir: Directory containing motion map videos
        saliency_dir: Directory containing saliency map videos
        viewport_dir: Base directory containing viewport data
        output_dir: Directory to save ranking JSON files
        video_id: Video identifier (e.g., 'sJxiPiAaB4k')
        user_id: User identifier (e.g., 'uid-3ba968b8-887c-460e-a5f2-86295957d731')
        test_id: Test identifier (e.g., 'test0')
    """
    motion_dir = Path(motion_dir)
    saliency_dir = Path(saliency_dir)
    viewport_dir = Path(viewport_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct paths
    motion_path = motion_dir / f"sparse_motion_{video_id}.mp4"
    saliency_path = saliency_dir / f"{video_id}_saliency_60s.mp4"
    
    # Find viewport CSV file
    viewport_base = viewport_dir / user_id / test_id
    viewport_files = list(viewport_base.glob(f"*{video_id}*/*_processed.csv"))
    
    if not viewport_files:
        print(f"❌ ERROR: No viewport data found for {video_id} in {viewport_base}")
        return
    
    viewport_path = viewport_files[0]
    
    print("=" * 60)
    print("VIEWPORT-AWARE TILE RANKING COMPUTATION")
    print("=" * 60)
    print(f"Video ID:    {video_id}")
    print(f"User ID:     {user_id}")
    print(f"Test ID:     {test_id}")
    print(f"Motion:      {motion_path}")
    print(f"Saliency:    {saliency_path}")
    print(f"Viewport:    {viewport_path}")
    print(f"Output:      {output_dir}")
    print("-" * 60)
    
    # Check if files exist
    if not motion_path.exists():
        print(f"❌ ERROR: Motion map not found: {motion_path}")
        return
    
    if not saliency_path.exists():
        print(f"❌ ERROR: Saliency map not found: {saliency_path}")
        return
    
    try:
        # Load viewport data
        print(f"\n📍 Loading viewport data...")
        viewport_data = load_viewport_data(viewport_path, target_fps=FPS)
        
        if not viewport_data:
            print(f"❌ ERROR: Failed to load viewport data")
            return
        
        print(f"  Loaded {len(viewport_data)} frames of viewport data")
        
        # Open videos
        print(f"\n🎥 Opening video files...")
        motion_cap = cv2.VideoCapture(str(motion_path))
        saliency_cap = cv2.VideoCapture(str(saliency_path))
        
        if not motion_cap.isOpened() or not saliency_cap.isOpened():
            print(f"❌ ERROR: Could not open video files")
            return
        
        # Get video properties
        frame_width = int(motion_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(motion_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(
            int(motion_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(saliency_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            len(viewport_data)
        )
        
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  Processing: {total_frames} frames")
        
        tile_boundaries = get_tile_boundaries(frame_height, frame_width)
        
        # Process frames
        print(f"\n⚙️  Processing frames...")
        rankings = {}
        frame_idx = 0
        
        while frame_idx < total_frames:
            motion_ret, motion_frame = motion_cap.read()
            saliency_ret, saliency_frame = saliency_cap.read()
            
            if not motion_ret or not saliency_ret or frame_idx not in viewport_data:
                break
            
            # Convert to grayscale if needed
            if len(motion_frame.shape) == 3:
                motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
            if len(saliency_frame.shape) == 3:
                saliency_frame = cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion and saliency scores
            motion_scores = calculate_motion_scores(motion_frame, tile_boundaries)
            saliency_scores = calculate_saliency_scores(saliency_frame, tile_boundaries)
            
            # Get viewport position and calculate viewport scores
            lat, lon = viewport_data[frame_idx]
            viewport_scores = calculate_viewport_scores(lat, lon, frame_width, frame_height)
            
            # Combine all scores and select top tiles
            top_tiles, combined_scores = select_tiles_viewport_aware(
                motion_scores, saliency_scores, viewport_scores, top_n=TOP_N_TILES
            )
            
            # Store rankings for this frame
            rankings[f"frame_{frame_idx}"] = {
                "top_tiles": top_tiles,
                "combined_scores": combined_scores.tolist(),
                "motion_scores": motion_scores.tolist(),
                "saliency_scores": saliency_scores.tolist(),
                "viewport_scores": viewport_scores.tolist(),
                "viewport_position": {
                    "latitude": lat,
                    "longitude": lon
                }
            }
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 300 == 0:
                print(f"  Progress: {frame_idx}/{total_frames} frames", end='\r')
        
        motion_cap.release()
        saliency_cap.release()
        
        print(f"\n  Processed {frame_idx} frames")
        
        # Save to JSON
        output_filename = f"{video_id}_{user_id}_{test_id}_rankings.json"
        output_path = output_dir / output_filename
        
        output_data = {
            "video_id": video_id,
            "user_id": user_id,
            "test_id": test_id,
            "total_frames": len(rankings),
            "fps": FPS,
            "duration_seconds": len(rankings) / FPS,
            "tile_config": {
                "rows": TILE_ROWS,
                "cols": TILE_COLS,
                "total_tiles": TOTAL_TILES,
                "top_n_tiles": TOP_N_TILES
            },
            "weights": {
                "motion": MOTION_WEIGHT,
                "saliency": SALIENCY_WEIGHT,
                "viewport": VIEWPORT_WEIGHT
            },
            "rankings": rankings
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Saved {len(rankings)} frame rankings to {output_path.name}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def precompute_viewport_aware_ranking(motion_path, saliency_path, viewport_path, output_dir, video_id, user_id, test_id):
    """
    Pre-compute viewport-aware tile rankings for a specific video and user.
    
    Args:
        motion_dir: Directory containing motion map videos
        saliency_dir: Directory containing saliency map videos
        viewport_dir: Base directory containing viewport data
        output_dir: Directory to save ranking JSON files
        video_id: Video identifier (e.g., 'sJxiPiAaB4k')
        user_id: User identifier (e.g., 'uid-3ba968b8-887c-460e-a5f2-86295957d731')
        test_id: Test identifier (e.g., 'test0')
    """
    motion_path = Path(motion_path)
    saliency_path = Path(saliency_path)
    viewport_path = Path(viewport_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VIEWPORT-AWARE TILE RANKING COMPUTATION")
    print("=" * 60)
    print(f"Video ID:    {video_id}")
    print(f"User ID:     {user_id}")
    print(f"Test ID:     {test_id}")
    print("-" * 60)
    
    # Check if files exist
    if not motion_path.exists():
        print(f"❌ ERROR: Motion map not found: {motion_path}")
        return
    
    if not saliency_path.exists():
        print(f"❌ ERROR: Saliency map not found: {saliency_path}")
        return
    
    try:
        # Load viewport data
        print(f"\n📍 Loading viewport data...")
        viewport_data = load_viewport_data(viewport_path, target_fps=FPS)
        
        if not viewport_data:
            print(f"❌ ERROR: Failed to load viewport data")
            return
        
        print(f"  Loaded {len(viewport_data)} frames of viewport data")
        
        # Open videos
        print(f"\n🎥 Opening video files...")
        motion_cap = cv2.VideoCapture(str(motion_path))
        saliency_cap = cv2.VideoCapture(str(saliency_path))
        
        if not motion_cap.isOpened() or not saliency_cap.isOpened():
            print(f"❌ ERROR: Could not open video files")
            return
        
        # Get video properties
        frame_width = int(motion_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(motion_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(
            int(motion_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(saliency_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            len(viewport_data)
        )
        
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  Processing: {total_frames} frames")
        
        tile_boundaries = get_tile_boundaries(frame_height, frame_width)
        
        # Process frames
        print(f"\n⚙️  Processing frames...")
        rankings = {}
        frame_idx = 0
        
        while frame_idx < total_frames and frame_idx in viewport_data:
            motion_ret, motion_frame = motion_cap.read()
            saliency_ret, saliency_frame = saliency_cap.read()
            
            if not motion_ret or not saliency_ret or frame_idx not in viewport_data:
                break
            
            # Convert to grayscale if needed
            if len(motion_frame.shape) == 3:
                motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
            if len(saliency_frame.shape) == 3:
                saliency_frame = cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion and saliency scores
            motion_scores = calculate_motion_scores(motion_frame, tile_boundaries)
            saliency_scores = calculate_saliency_scores(saliency_frame, tile_boundaries)
            
            # Get viewport position and calculate viewport scores
            lat, lon = viewport_data[frame_idx]
            viewport_scores = calculate_viewport_scores(lat, lon, frame_width, frame_height)
            
            # Combine all scores and select top tiles
            top_tiles, combined_scores = select_tiles_viewport_aware(
                motion_scores, saliency_scores, viewport_scores, top_n=TOP_N_TILES
            )
            
            # Store rankings for this frame
            rankings[f"frame_{frame_idx}"] = {
                "top_tiles": top_tiles,
                "combined_scores": combined_scores.tolist(),
                "motion_scores": motion_scores.tolist(),
                "saliency_scores": saliency_scores.tolist(),
                "viewport_scores": viewport_scores.tolist(),
                "viewport_position": {
                    "latitude": lat,
                    "longitude": lon
                }
            }
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 300 == 0:
                print(f"  Progress: {frame_idx}/{total_frames} frames", end='\r')
        
        motion_cap.release()
        saliency_cap.release()
        
        print(f"\n  Processed {frame_idx} frames")
        
        # Save to JSON
        output_filename = f"{video_id}_{user_id}_{test_id}_rankings.json"
        output_path = output_dir / output_filename
        
        output_data = {
            "video_id": video_id,
            "user_id": user_id,
            "test_id": test_id,
            "total_frames": len(rankings),
            "fps": FPS,
            "duration_seconds": len(rankings) / FPS,
            "tile_config": {
                "rows": TILE_ROWS,
                "cols": TILE_COLS,
                "total_tiles": TOTAL_TILES,
                "top_n_tiles": TOP_N_TILES
            },
            "weights": {
                "motion": MOTION_WEIGHT,
                "saliency": SALIENCY_WEIGHT,
                "viewport": VIEWPORT_WEIGHT
            },
            "rankings": rankings
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Saved {len(rankings)} frame rankings to {output_path.name}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Starting Viewport-Aware Tile Ranking Computation\n")
    
    # Example configuration - MODIFY THESE
    VIDEO_ID = "sJxiPiAaB4k"
    USER_ID = "uid-3ba968b8-887c-460e-a5f2-86295957d731"
    TEST_ID = "test0"
    MOTION_PATH = "/Users/eunicechoi04/Downloads/videoabr/output/motion_maps/sparse_motion_sJxiPiAaB4k.mp4"
    SALIENCY_PATH = "/Users/eunicechoi04/Downloads/videoabr/output/saliency_videos_60s/sJxiPiAaB4k_saliency_60s.mp4"
    VIEWPORT_PATH = "/Users/eunicechoi04/Downloads/videoabr/360_Video_analysis/data/uid-3ba968b8-887c-460e-a5f2-86295957d731/test0/Paris-sJxiPiAaB4k/Paris-sJxiPiAaB4k_0_processed.csv"
    precompute_viewport_aware_ranking(
        motion_path=MOTION_PATH,
        saliency_path=SALIENCY_PATH,
        viewport_path=VIEWPORT_PATH,
        output_dir=OUTPUT_DIR,
        video_id=VIDEO_ID,
        user_id=USER_ID,
        test_id=TEST_ID
    )


    # Run computation
    # precompute_viewport_aware_rankings(
    #     motion_dir=MOTION_DIR,
    #     saliency_dir=SALIENCY_DIR,
    #     viewport_dir=VIEWPORT_DIR,
    #     output_dir=OUTPUT_DIR,
    #     video_id=VIDEO_ID,
    #     user_id=USER_ID,
    #     test_id=TEST_ID
    # )
    
    print("\n✨ Done! Viewport-aware rankings are ready.")
    print(f"   Output: {OUTPUT_DIR}")
