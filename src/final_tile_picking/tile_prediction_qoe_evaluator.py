"""
QoE Evaluation for Predictive Tiling in 360° Video Streaming

Compares two approaches:
1. BASELINE: Full 360° sphere at high quality (all 24 tiles at 5000 Kbps)
2. PREDICTIVE TILING: Predicted tiles at high quality, rest at low quality

Metrics:
- QoE Score (quality reward - rebuffering penalty - viewport miss penalty)
- Rebuffering time and ratio
- Viewport quality hit rate (prediction accuracy)
- Bandwidth savings
- Average quality in viewport
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ==========================================
# CONFIGURATION
# ==========================================
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
SEGMENT_DURATION = 1.0  # seconds

# Bitrate configuration
BITRATE_LOW_KBPS = 500      # Background tiles
BITRATE_HIGH_KBPS = 5000    # Viewport tiles

# QoE weights (based on Pensieve/Puffer research)
QOE_QUALITY_WEIGHT = 1.0         # Linear quality reward
QOE_REBUF_PENALTY = 4.3          # Rebuffering is 4.3x worse than quality drop
QOE_VIEWPORT_MISS_PENALTY = 2.0  # Penalty per viewport tile at low quality

# Buffer parameters
BUFFER_INIT = 0.0
BUFFER_MAX = 60.0


# ==========================================
# CORE SIMULATION ENGINE
# ==========================================

def simulate_baseline_streaming(
    video_name: str,
    num_segments: int,
    network_trace_mbps: List[float],
    actual_viewport_tiles: Optional[List[List[int]]] = None
) -> Dict:
    """
    Baseline: Download entire 360° sphere at high quality

    This represents perfect quality everywhere, but uses maximum bandwidth.
    """

    buffer_level = BUFFER_INIT
    total_rebuf_time = 0.0
    total_bits_downloaded = 0

    segment_details = []

    for seg_idx in range(num_segments):
        bandwidth_mbps = network_trace_mbps[seg_idx % len(network_trace_mbps)]

        # Download ALL tiles at high quality
        segment_size_bits = TOTAL_TILES * BITRATE_HIGH_KBPS * 1000 * SEGMENT_DURATION
        total_bits_downloaded += segment_size_bits

        # Download simulation
        segment_size_mb = segment_size_bits / (8 * 1_000_000)
        download_time = segment_size_mb / max(bandwidth_mbps, 0.01)

        # Buffer update
        rebuf_time = 0
        if download_time > SEGMENT_DURATION:
            deficit = download_time - SEGMENT_DURATION
            if buffer_level >= deficit:
                buffer_level -= deficit
            else:
                rebuf_time = deficit - buffer_level
                total_rebuf_time += rebuf_time
                buffer_level = 0
        else:
            buffer_level += (SEGMENT_DURATION - download_time)
            buffer_level = min(buffer_level, BUFFER_MAX)

        segment_details.append({
            'segment': seg_idx,
            'bandwidth_mbps': bandwidth_mbps,
            'buffer_level': buffer_level,
            'rebuf_time': rebuf_time,
            'download_time': download_time
        })

    # Metrics
    video_duration = num_segments * SEGMENT_DURATION
    total_mb_downloaded = total_bits_downloaded / (8 * 1_000_000)

    # QoE: All viewport tiles are always high quality, so no viewport miss penalty
    avg_viewport_quality = BITRATE_HIGH_KBPS / 1000.0  # Mbps
    qoe_score = (
        QOE_QUALITY_WEIGHT * avg_viewport_quality * num_segments -
        QOE_REBUF_PENALTY * total_rebuf_time
    )

    return {
        'approach': 'baseline_full_quality',
        'video_name': video_name,
        'num_segments': num_segments,
        'video_duration': video_duration,

        # QoE metrics
        'qoe_score': qoe_score,
        'rebuf_time': total_rebuf_time,
        'rebuf_ratio': total_rebuf_time / video_duration,

        # Quality metrics
        'avg_viewport_quality_mbps': avg_viewport_quality,
        'viewport_hit_rate': 1.0,  # Perfect - all tiles are high quality
        'viewport_misses': 0,

        # Bandwidth metrics
        'total_mb_downloaded': total_mb_downloaded,
        'bandwidth_savings': 0.0,  # Baseline uses 100% bandwidth
        'avg_bandwidth_mbps': total_mb_downloaded * 8 / video_duration,

        'segments': segment_details
    }


def simulate_predictive_tiling(
    video_name: str,
    num_segments: int,
    network_trace_mbps: List[float],
    predicted_tiles: List[List[int]],  # Your tile predictions
    actual_viewport_tiles: List[List[int]],  # Ground truth
    num_high_quality_tiles: int = 4
) -> Dict:
    """
    Predictive Tiling: Use your tile predictor to select which tiles to download at high quality

    Args:
        video_name: Name of video
        num_segments: Number of 1-second segments
        network_trace_mbps: Bandwidth trace
        predicted_tiles: Your predictions - list of tile indices per segment
        actual_viewport_tiles: Ground truth viewport tiles per segment
        num_high_quality_tiles: How many tiles to download at high quality
    """

    buffer_level = BUFFER_INIT
    total_rebuf_time = 0.0
    total_bits_downloaded = 0

    viewport_hits = 0
    viewport_misses = 0
    viewport_quality_sum = 0.0

    segment_details = []

    for seg_idx in range(num_segments):
        bandwidth_mbps = network_trace_mbps[seg_idx % len(network_trace_mbps)]

        # Get your tile predictions for this segment
        if seg_idx < len(predicted_tiles):
            high_quality_tiles = predicted_tiles[seg_idx][:num_high_quality_tiles]
        else:
            # Fallback if predictions are missing
            high_quality_tiles = [10, 11, 13, 14]  # Center tiles

        low_quality_tiles = [i for i in range(TOTAL_TILES) if i not in high_quality_tiles]

        # Calculate download size
        num_high = len(high_quality_tiles)
        num_low = len(low_quality_tiles)

        segment_size_bits = (
            num_high * BITRATE_HIGH_KBPS * 1000 * SEGMENT_DURATION +
            num_low * BITRATE_LOW_KBPS * 1000 * SEGMENT_DURATION
        )
        total_bits_downloaded += segment_size_bits

        # Download simulation
        segment_size_mb = segment_size_bits / (8 * 1_000_000)
        download_time = segment_size_mb / max(bandwidth_mbps, 0.01)

        # Buffer update
        rebuf_time = 0
        if download_time > SEGMENT_DURATION:
            deficit = download_time - SEGMENT_DURATION
            if buffer_level >= deficit:
                buffer_level -= deficit
            else:
                rebuf_time = deficit - buffer_level
                total_rebuf_time += rebuf_time
                buffer_level = 0
        else:
            buffer_level += (SEGMENT_DURATION - download_time)
            buffer_level = min(buffer_level, BUFFER_MAX)

        # Evaluate prediction accuracy
        if seg_idx < len(actual_viewport_tiles):
            actual_viewed = set(actual_viewport_tiles[seg_idx])
            predicted_high = set(high_quality_tiles)

            hits = len(actual_viewed & predicted_high)
            misses = len(actual_viewed - predicted_high)

            viewport_hits += hits
            viewport_misses += misses

            # Calculate average quality in viewport
            # Tiles that were predicted correctly get high quality, rest get low
            viewport_quality = (
                hits * BITRATE_HIGH_KBPS +
                misses * BITRATE_LOW_KBPS
            ) / max(len(actual_viewed), 1)
            viewport_quality_sum += viewport_quality

        segment_details.append({
            'segment': seg_idx,
            'bandwidth_mbps': bandwidth_mbps,
            'buffer_level': buffer_level,
            'rebuf_time': rebuf_time,
            'download_time': download_time,
            'predicted_tiles': high_quality_tiles,
            'actual_tiles': actual_viewport_tiles[seg_idx] if seg_idx < len(actual_viewport_tiles) else []
        })

    # Metrics
    video_duration = num_segments * SEGMENT_DURATION
    total_mb_downloaded = total_bits_downloaded / (8 * 1_000_000)

    # Baseline comparison
    baseline_bits = TOTAL_TILES * BITRATE_HIGH_KBPS * 1000 * video_duration
    baseline_mb = baseline_bits / (8 * 1_000_000)
    bandwidth_savings = 1.0 - (total_mb_downloaded / baseline_mb)

    # Viewport metrics
    viewport_hit_rate = viewport_hits / max(viewport_hits + viewport_misses, 1)
    avg_viewport_quality = viewport_quality_sum / num_segments / 1000.0  # Convert to Mbps

    # QoE score
    qoe_score = (
        QOE_QUALITY_WEIGHT * avg_viewport_quality * num_segments -
        QOE_REBUF_PENALTY * total_rebuf_time -
        QOE_VIEWPORT_MISS_PENALTY * viewport_misses
    )

    return {
        'approach': 'predictive_tiling',
        'video_name': video_name,
        'num_segments': num_segments,
        'video_duration': video_duration,

        # QoE metrics
        'qoe_score': qoe_score,
        'rebuf_time': total_rebuf_time,
        'rebuf_ratio': total_rebuf_time / video_duration,

        # Quality metrics
        'avg_viewport_quality_mbps': avg_viewport_quality,
        'viewport_hit_rate': viewport_hit_rate,
        'viewport_hits': viewport_hits,
        'viewport_misses': viewport_misses,

        # Bandwidth metrics
        'total_mb_downloaded': total_mb_downloaded,
        'bandwidth_savings': bandwidth_savings,
        'avg_bandwidth_mbps': total_mb_downloaded * 8 / video_duration,

        'segments': segment_details
    }


# ==========================================
# BATCH EVALUATION WITH GROUND TRUTH
# ==========================================

def load_viewport_ground_truth(user_trace_file: str) -> List[List[int]]:
    """
    Load ground truth viewport tiles from processed user trace CSV

    Expected CSV format:
    - Contains 'tile_id' column with tile index user was viewing
    - One row per frame or timestamp

    Returns list of tile indices per second
    """
    df = pd.read_csv(user_trace_file)

    if 'tile_id' not in df.columns:
        raise ValueError(f"CSV missing 'tile_id' column: {user_trace_file}")

    # Group by second and get unique tiles viewed in each second
    # Assuming 'timestamp' or 'frame_id' column exists
    if 'timestamp' in df.columns:
        df['second'] = df['timestamp'].astype(int)
    elif 'frame_id' in df.columns:
        # Assume 30 FPS
        df['second'] = df['frame_id'] // 30
    else:
        # Fallback: just group by row index
        df['second'] = df.index // 30

    viewport_per_second = []
    for second in sorted(df['second'].unique()):
        tiles_in_second = df[df['second'] == second]['tile_id'].unique().tolist()
        viewport_per_second.append(tiles_in_second)

    return viewport_per_second


def run_comparison_evaluation(
    video_name: str,
    user_trace_file: str,  # Ground truth viewport
    predicted_tiles: List[List[int]],  # Your tile predictions
    network_traces: Dict[str, List[float]],
    output_dir: str = "qoe_results"
) -> pd.DataFrame:
    """
    Run full comparison: Baseline vs Predictive Tiling across multiple network conditions

    Args:
        video_name: Name of the video
        user_trace_file: Path to CSV with ground truth viewport tiles
        predicted_tiles: Your tile predictions (list of tile lists per segment)
        network_traces: Dict of network trace name -> bandwidth list (Mbps)
        output_dir: Where to save results

    Returns:
        DataFrame with comparison results
    """

    # Load ground truth
    print(f"Loading ground truth from: {user_trace_file}")
    actual_viewport = load_viewport_ground_truth(user_trace_file)
    num_segments = min(len(predicted_tiles), len(actual_viewport))

    print(f"Video: {video_name}")
    print(f"Segments: {num_segments}")
    print(f"Network traces: {list(network_traces.keys())}")
    print("="*60)

    results = []

    for trace_name, trace_bandwidth in network_traces.items():
        print(f"\nNetwork Condition: {trace_name}")
        print(f"  Avg Bandwidth: {np.mean(trace_bandwidth):.2f} Mbps")

        # 1. Baseline: Full quality
        print("  Running baseline (full quality)...")
        baseline_result = simulate_baseline_streaming(
            video_name=video_name,
            num_segments=num_segments,
            network_trace_mbps=trace_bandwidth,
            actual_viewport_tiles=actual_viewport
        )
        baseline_result['network_trace'] = trace_name
        results.append({k: v for k, v in baseline_result.items() if k != 'segments'})

        print(f"    QoE: {baseline_result['qoe_score']:.2f}")
        print(f"    Rebuffer: {baseline_result['rebuf_time']:.2f}s")
        print(f"    Bandwidth: {baseline_result['total_mb_downloaded']:.2f} MB")

        # 2. Predictive Tiling: Your approach
        print("  Running predictive tiling...")
        predictive_result = simulate_predictive_tiling(
            video_name=video_name,
            num_segments=num_segments,
            network_trace_mbps=trace_bandwidth,
            predicted_tiles=predicted_tiles,
            actual_viewport_tiles=actual_viewport
        )
        predictive_result['network_trace'] = trace_name
        results.append({k: v for k, v in predictive_result.items() if k != 'segments'})

        print(f"    QoE: {predictive_result['qoe_score']:.2f}")
        print(f"    Rebuffer: {predictive_result['rebuf_time']:.2f}s")
        print(f"    Bandwidth: {predictive_result['total_mb_downloaded']:.2f} MB")
        print(f"    Savings: {predictive_result['bandwidth_savings']:.1%}")
        print(f"    Hit Rate: {predictive_result['viewport_hit_rate']:.1%}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_qoe_comparison.csv")
    df.to_csv(output_file, index=False)

    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    # Print comparison table
    comparison_cols = ['approach', 'network_trace', 'qoe_score', 'rebuf_time',
                      'avg_viewport_quality_mbps', 'bandwidth_savings', 'viewport_hit_rate']
    print(df[comparison_cols].to_string(index=False))

    print(f"\nResults saved to: {output_file}")

    return df


# ==========================================
# INTEGRATION WITH EXISTING CODE
# ==========================================

def integrate_with_content_aware_simulator(
    motion_video_path: str,
    saliency_video_path: str,
    user_trace_file: str,
    network_traces: Dict[str, List[float]],
    output_dir: str = "qoe_results"
) -> pd.DataFrame:
    """
    Use existing motion/saliency videos to generate predictions,
    then evaluate QoE

    This integrates with your existing content-aware ABR simulator
    """

    from final_tile_picking.content_aware_abr_simulator import (
        get_tile_boundaries, calculate_motion_scores, calculate_saliency_scores,
        select_tiles_content_aware, FPS, SEGMENT_DURATION, TOP_N_TILES
    )

    # Open videos
    motion_cap = cv2.VideoCapture(motion_video_path)
    saliency_cap = cv2.VideoCapture(saliency_video_path)

    if not motion_cap.isOpened() or not saliency_cap.isOpened():
        raise ValueError("Could not open motion/saliency videos")

    # Get properties
    frame_width = int(motion_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(motion_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tile_boundaries = get_tile_boundaries(frame_height, frame_width)

    # Generate predictions segment by segment
    predicted_tiles = []
    frames_per_segment = int(FPS * SEGMENT_DURATION)

    print("Generating tile predictions from content analysis...")
    while True:
        motion_ret, motion_frame = motion_cap.read()
        saliency_ret, saliency_frame = saliency_cap.read()

        if not motion_ret or not saliency_ret:
            break

        # Convert to grayscale
        if len(motion_frame.shape) == 3:
            motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
        if len(saliency_frame.shape) == 3:
            saliency_frame = cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2GRAY)

        # Calculate scores
        motion_scores = calculate_motion_scores(motion_frame, tile_boundaries)
        saliency_scores = calculate_saliency_scores(saliency_frame, tile_boundaries)

        # Select tiles
        high_tiles, _ = select_tiles_content_aware(motion_scores, saliency_scores, top_n=TOP_N_TILES)
        predicted_tiles.append(high_tiles)

        # Skip to next segment
        for _ in range(frames_per_segment - 1):
            motion_cap.read()
            saliency_cap.read()

    motion_cap.release()
    saliency_cap.release()

    print(f"Generated {len(predicted_tiles)} tile predictions")

    # Extract video name
    video_name = Path(motion_video_path).stem.replace("sparse_motion_", "")

    # Run comparison evaluation
    return run_comparison_evaluation(
        video_name=video_name,
        user_trace_file=user_trace_file,
        predicted_tiles=predicted_tiles,
        network_traces=network_traces,
        output_dir=output_dir
    )


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Example: Evaluate one video with multiple network conditions

    # Network traces (you can load these from your data/traces/ directory)
    network_traces = {
        'high_10mbps': [10.0] * 60,
        'medium_5mbps': [5.0] * 60,
        'low_2mbps': [2.0] * 60,
        'variable': [3, 5, 8, 4, 6, 2, 7, 9, 3, 5] * 6
    }

    # Example tile predictions (replace with your actual predictions)
    # Each inner list is the predicted high-quality tiles for that second
    predicted_tiles = [
        [10, 11, 13, 14],  # Second 0
        [10, 11, 13, 14],  # Second 1
        [11, 12, 14, 15],  # Second 2
        # ... continue for all seconds
    ] * 20  # 60 seconds total

    # Ground truth viewport (load from your processed user traces)
    user_trace_file = "360_Video_analysis/data/uid-1/test0/Diving-2OzlksZBTiA_0_processed.csv"

    # Run comparison
    results = run_comparison_evaluation(
        video_name="Diving-2OzlksZBTiA",
        user_trace_file=user_trace_file,
        predicted_tiles=predicted_tiles,
        network_traces=network_traces,
        output_dir="qoe_results"
    )

    print("\nDone! Check qoe_results/ directory for detailed results.")
