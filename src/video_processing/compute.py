import os
import sys
import json
import cv2
import numpy as np
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

# Weighting factors for content-aware tile selection
MOTION_WEIGHT = 0.05
SALIENCY_WEIGHT = 0.95
TOP_N_TILES = 9  # Number of tiles to mark as high priority

# Paths
MOTION_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/motion_maps"
SALIENCY_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/saliency_videos_60s"
OUTPUT_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/05_95_tile_rankings"


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
        # Motion intensity = percentage of white pixels in tile
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
        # Average saliency in tile
        saliency_score = np.mean(tile_region) / 255.0
        scores.append(saliency_score)
    return np.array(scores)


def select_tiles_content_aware(motion_scores, saliency_scores, top_n=TOP_N_TILES):
    """
    Combine motion and saliency scores to select top N tiles for high quality.
    Returns: List of tile indices + combined scores
    """
    # Combine scores with weights
    combined_scores = (MOTION_WEIGHT * motion_scores +
                      SALIENCY_WEIGHT * saliency_scores)

    # Get indices of top N tiles
    top_tile_indices = np.argsort(combined_scores)[-top_n:]

    return top_tile_indices.tolist(), combined_scores


# ==========================================
# MAIN PRE-COMPUTATION FUNCTION
# ==========================================
def precompute_tile_rankings(motion_dir, saliency_dir, output_dir):
    """
    Pre-analyze all videos and save tile rankings for each frame.
    
    Args:
        motion_dir: Directory containing motion map videos
        saliency_dir: Directory containing saliency map videos
        output_dir: Directory to save ranking JSON files
    
    Saves one JSON file per video: {video_id}_rankings.json
    Format: {
        "video_id": "video1",
        "total_frames": 1800,
        "fps": 30,
        "duration_seconds": 60,
        "tile_config": {...},
        "rankings": {
            "frame_0": {
                "top_tiles": [23, 17, 16, 11],
                "combined_scores": [0.8, 0.75, ...],
                "motion_scores": [0.5, 0.4, ...],
                "saliency_scores": [0.3, 0.35, ...]
            },
            "frame_1": { ... },
            ...
        }
    }
    """
    motion_dir = Path(motion_dir)
    saliency_dir = Path(saliency_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    motion_videos = [list(motion_dir.glob("sparse_motion_*.mp4"))[-1]]
    print(motion_videos)
    if len(motion_videos) == 0:
        print(f"ERROR: No motion videos found in {motion_dir}")
        return
    
    print("=" * 60)
    print("PRE-COMPUTING TILE RANKINGS")
    print("=" * 60)
    print(f"Motion videos:  {motion_dir}")
    print(f"Saliency videos: {saliency_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Found {len(motion_videos)} motion map videos")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for motion_path in motion_videos:
        video_id = motion_path.stem.replace("sparse_motion_", "")
        saliency_path = saliency_dir / f"{video_id}_saliency_60s.mp4"
        
        if not saliency_path.exists():
            print(f"⚠️  WARNING: Saliency map not found for {video_id}, skipping...")
            failed += 1
            continue
        
        print(f"\n📹 Processing: {video_id}")
        
        try:
            # Open videos
            motion_cap = cv2.VideoCapture(str(motion_path))
            saliency_cap = cv2.VideoCapture(str(saliency_path))
            
            if not motion_cap.isOpened() or not saliency_cap.isOpened():
                print(f"  ❌ ERROR: Could not open video files")
                failed += 1
                continue
            
            # Get video properties
            frame_width = int(motion_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(motion_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(motion_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"  Resolution: {frame_width}x{frame_height}")
            print(f"  Total frames: {total_frames}")
            
            tile_boundaries = get_tile_boundaries(frame_height, frame_width)
            
            # Store rankings for all frames (30 FPS)
            rankings = {}
            frame_idx = 0
            
            while True:
                motion_ret, motion_frame = motion_cap.read()
                saliency_ret, saliency_frame = saliency_cap.read()
                
                if not motion_ret or not saliency_ret:
                    break
                
                # Convert to grayscale if needed
                if len(motion_frame.shape) == 3:
                    motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
                if len(saliency_frame.shape) == 3:
                    saliency_frame = cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate scores
                motion_scores = calculate_motion_scores(motion_frame, tile_boundaries)
                saliency_scores = calculate_saliency_scores(saliency_frame, tile_boundaries)
                
                # Get top tiles
                top_tiles, combined_scores = select_tiles_content_aware(
                    motion_scores, saliency_scores, top_n=TOP_N_TILES
                )
                
                # Store rankings for EVERY frame
                rankings[f"frame_{frame_idx}"] = {
                    "top_tiles": top_tiles,
                    "combined_scores": combined_scores.tolist(),
                    "motion_scores": motion_scores.tolist(),
                    "saliency_scores": saliency_scores.tolist()
                }
                
                frame_idx += 1
                
                # Progress indicator
                if frame_idx % 300 == 0:  # Every 10 seconds
                    print(f"  Progress: {frame_idx}/{total_frames} frames", end='\r')
            
            motion_cap.release()
            saliency_cap.release()
            
            # Save to JSON
            output_path = output_dir / f"{video_id}_rankings.json"
            
            output_data = {
                "video_id": video_id,
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
                    "saliency": SALIENCY_WEIGHT
                },
                "rankings": rankings
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  ✅ Saved {len(rankings)} frame rankings to {output_path.name}")
            successful += 1
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful} videos")
    print(f"❌ Failed:     {failed} videos")
    print(f"📁 Output:     {output_dir}")
    print("=" * 60)


# ==========================================
# UTILITY: Load pre-computed rankings
# ==========================================
def load_precomputed_rankings(video_id, rankings_dir):
    """
    Load pre-computed tile rankings for a video.
    
    Returns: Dictionary with rankings, or None if not found
    """
    rankings_path = Path(rankings_dir) / f"{video_id}_rankings.json"
    
    if not rankings_path.exists():
        print(f"WARNING: Rankings not found for {video_id}")
        return None
    
    with open(rankings_path, 'r') as f:
        return json.load(f)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Starting Tile Ranking Pre-computation\n")
    
    # Check if directories exist
    if not os.path.exists(MOTION_DIR):
        print(f"❌ ERROR: Motion directory not found: {MOTION_DIR}")
        sys.exit(1)
    
    if not os.path.exists(SALIENCY_DIR):
        print(f"❌ ERROR: Saliency directory not found: {SALIENCY_DIR}")
        sys.exit(1)
    
    # Run pre-computation
    precompute_tile_rankings(
        motion_dir=MOTION_DIR,
        saliency_dir=SALIENCY_DIR,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✨ Done! Rankings are ready for simulation.")
    print(f"   Run your simulation script with rankings from: {OUTPUT_DIR}")