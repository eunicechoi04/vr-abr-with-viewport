import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
BITRATE_LOW_KBPS = 500
BITRATE_HIGH_KBPS = 5000
SEGMENT_DURATION = 1.0  # seconds
FPS = 30  # frames per second

# Weighting factors for content-aware tile selection
MOTION_WEIGHT = 0.5
SALIENCY_WEIGHT = 0.5
TOP_N_TILES = 4  # Number of tiles to download at high quality


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
            row_start = row * tile_height
            row_end = (row + 1) * tile_height if row < TILE_ROWS - 1 else frame_height
            col_start = col * tile_width
            col_end = (col + 1) * tile_width if col < TILE_COLS - 1 else frame_width
            tiles.append((row_start, row_end, col_start, col_end))

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
    Returns: List of tile indices to download at high quality
    """
    # Combine scores with weights
    combined_scores = (MOTION_WEIGHT * motion_scores +
                      SALIENCY_WEIGHT * saliency_scores)

    # Get indices of top N tiles
    top_tile_indices = np.argsort(combined_scores)[-top_n:]

    return top_tile_indices.tolist(), combined_scores


# ==========================================
# SIMULATION FUNCTIONS
# ==========================================
def run_simulation_on_video(video_name, motion_video_path, saliency_video_path,
                           network_trace_mbps, output_results=None):
    """
    Simulate ABR streaming for a single video using motion and saliency maps.

    Args:
        video_name: Name of the video
        motion_video_path: Path to motion map video
        saliency_video_path: Path to saliency map video
        network_trace_mbps: List of bandwidth values in Mbps (cycles through)
        output_results: Optional dict to store detailed frame-by-frame results

    Returns:
        Dictionary with simulation metrics
    """
    print(f"Processing: {video_name}")

    # Open motion and saliency videos
    motion_cap = cv2.VideoCapture(motion_video_path)
    saliency_cap = cv2.VideoCapture(saliency_video_path)

    if not motion_cap.isOpened() or not saliency_cap.isOpened():
        print(f"  ERROR: Could not open video files")
        return None

    # Get video properties
    frame_width = int(motion_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(motion_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(motion_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate tile boundaries
    tile_boundaries = get_tile_boundaries(frame_height, frame_width)

    # Simulation state
    buffer_level = 0.0
    total_stall_time = 0.0
    total_bits_downloaded = 0

    # Process video segment by segment (1 second = 30 frames)
    segment_idx = 0
    frames_per_segment = int(FPS * SEGMENT_DURATION)

    if output_results is not None:
        output_results['segments'] = []

    while True:
        # Read one frame from each video (we'll analyze every Nth frame per segment)
        # For simplicity, analyze the first frame of each segment
        motion_ret, motion_frame = motion_cap.read()
        saliency_ret, saliency_frame = saliency_cap.read()

        if not motion_ret or not saliency_ret:
            break

        # Convert to grayscale if needed
        if len(motion_frame.shape) == 3:
            motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
        if len(saliency_frame.shape) == 3:
            saliency_frame = cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2GRAY)

        # Calculate content scores for each tile
        motion_scores = calculate_motion_scores(motion_frame, tile_boundaries)
        saliency_scores = calculate_saliency_scores(saliency_frame, tile_boundaries)

        # Select tiles for high quality download
        high_quality_tiles, combined_scores = select_tiles_content_aware(
            motion_scores, saliency_scores, top_n=TOP_N_TILES
        )

        # Calculate download size for this segment
        num_high = len(high_quality_tiles)
        num_low = TOTAL_TILES - num_high

        # Size in bits for this segment
        segment_size_bits = (num_high * BITRATE_HIGH_KBPS * 1000 * SEGMENT_DURATION +
                           num_low * BITRATE_LOW_KBPS * 1000 * SEGMENT_DURATION)

        total_bits_downloaded += segment_size_bits

        # Network simulation
        bandwidth_mbps = network_trace_mbps[segment_idx % len(network_trace_mbps)]
        bandwidth_bps = bandwidth_mbps * 1_000_000

        download_time = segment_size_bits / bandwidth_bps

        # Buffer simulation
        if download_time > SEGMENT_DURATION:
            # Download took too long - buffer drains
            deficit = download_time - SEGMENT_DURATION
            if buffer_level >= deficit:
                buffer_level -= deficit
            else:
                stall = deficit - buffer_level
                total_stall_time += stall
                buffer_level = 0
        else:
            # Download was fast - buffer grows
            buffer_level += (SEGMENT_DURATION - download_time)

        # Store detailed results if requested
        if output_results is not None:
            output_results['segments'].append({
                'segment': segment_idx,
                'high_quality_tiles': high_quality_tiles,
                'motion_scores': motion_scores.tolist(),
                'saliency_scores': saliency_scores.tolist(),
                'combined_scores': combined_scores.tolist(),
                'bandwidth_mbps': bandwidth_mbps,
                'download_time': download_time,
                'buffer_level': buffer_level
            })

        # Skip to next segment (advance frames)
        for _ in range(frames_per_segment - 1):
            motion_cap.read()
            saliency_cap.read()

        segment_idx += 1

    motion_cap.release()
    saliency_cap.release()

    # Calculate metrics
    video_duration = segment_idx * SEGMENT_DURATION
    baseline_bits = TOTAL_TILES * BITRATE_HIGH_KBPS * 1000 * video_duration
    bandwidth_savings = 1.0 - (total_bits_downloaded / baseline_bits) if baseline_bits > 0 else 0
    stall_ratio = total_stall_time / video_duration if video_duration > 0 else 0

    print(f"  Segments: {segment_idx}")
    print(f"  Bandwidth Savings: {bandwidth_savings:.2%}")
    print(f"  Stall Time: {total_stall_time:.2f}s ({stall_ratio:.2%})")
    print(f"  Total Downloaded: {total_bits_downloaded / 1_000_000:.2f} Mb")

    return {
        'video_name': video_name,
        'segments': segment_idx,
        'bandwidth_savings': bandwidth_savings,
        'total_stall_time': total_stall_time,
        'stall_ratio': stall_ratio,
        'total_bits_mb': total_bits_downloaded / 1_000_000,
        'baseline_bits_mb': baseline_bits / 1_000_000
    }


def run_batch_simulation(motion_dir, saliency_dir, network_traces, output_csv=None):
    """
    Run simulations on all videos with different network traces.

    Args:
        motion_dir: Directory containing motion map videos
        saliency_dir: Directory containing saliency map videos
        network_traces: Dictionary mapping trace names to bandwidth lists
                       e.g., {'stable': [5, 5, 5, ...], 'variable': [3, 5, 8, ...]}
        output_csv: Optional path to save results CSV

    Returns:
        DataFrame with results for all videos and network conditions
    """
    motion_dir = Path(motion_dir)
    saliency_dir = Path(saliency_dir)

    # Find all motion map videos
    motion_videos = list(motion_dir.glob("sparse_motion_*.mp4"))

    if len(motion_videos) == 0:
        print("ERROR: No motion map videos found!")
        return None

    print(f"Found {len(motion_videos)} motion map videos")
    print(f"Testing with {len(network_traces)} network conditions")
    print("-" * 60)

    results = []

    for motion_path in motion_videos:
        # Extract video name (e.g., "2OzlksZBTiA" from "sparse_motion_2OzlksZBTiA.mp4")
        video_id = motion_path.stem.replace("sparse_motion_", "")

        # Find corresponding saliency video
        saliency_path = saliency_dir / f"{video_id}_saliency_60s.mp4"
        print(saliency_path)
        if not saliency_path.exists():
            print(f"WARNING: Saliency map not found for {video_id}, skipping...")
            continue


        # Run simulation with each network trace
        for trace_name, trace_bandwidth in network_traces.items():
            print(f"\n{video_id} - {trace_name}")
            result = run_simulation_on_video(
                video_name=f"{video_id}_{trace_name}",
                motion_video_path=str(motion_path),
                saliency_video_path=str(saliency_path),
                network_trace_mbps=trace_bandwidth
            )

            if result:
                result['video_id'] = video_id
                result['network_trace'] = trace_name
                results.append(result)

    # Create results dataframe
    if len(results) == 0:
        print("ERROR: No successful simulations!")
        return None

    df = pd.DataFrame(results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Average Bandwidth Savings: {df['bandwidth_savings'].mean():.2%}")
    print(f"  Average Stall Ratio:       {df['stall_ratio'].mean():.2%}")
    print(f"  Average Stall Time:        {df['total_stall_time'].mean():.2f}s")

    print(f"\nBy Network Condition:")
    for trace_name in df['network_trace'].unique():
        trace_df = df[df['network_trace'] == trace_name]
        print(f"\n  {trace_name}:")
        print(f"    Bandwidth Savings: {trace_df['bandwidth_savings'].mean():.2%}")
        print(f"    Stall Ratio:       {trace_df['stall_ratio'].mean():.2%}")

    # Save results if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    return df