"""
Run Tile-based ABR Experiments - mimics 6.5820_pset_3/scripts/run_exps.py

This orchestrates multiple experiment runs:
- Iterates through all users who viewed a video
- Tests across multiple network conditions
- Compares baseline vs predictive tiling approaches
- Saves results for analysis
"""

import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing.pool import ThreadPool
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from final_tile_picking.tile_abr_env import (
    run_single_experiment,
    TOTAL_TILES,
    BITRATE_HIGH_KBPS,
    BITRATE_LOW_KBPS,
    SEGMENT_DURATION,
    TILE_ROWS,
    TILE_COLS
)


def load_network_trace(trace_path: str, duration_seconds: int = 60) -> List[float]:
    """
    Load network trace from file.
    
    Supports:
    - .dat files (Kbps values)
    - .mahi files (Mbps values)
    - .log files (Mbps values)
    """
    trace_path = Path(trace_path)
    
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    
    with open(trace_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    bandwidths = []
    for line in lines:
        try:
            bw = float(line)
            # Convert .dat from Kbps to Mbps
            if trace_path.suffix == '.dat':
                bw = bw / 1000.0
            bandwidths.append(bw)
        except ValueError:
            continue
    
    if len(bandwidths) == 0:
        raise ValueError(f"No valid bandwidth values: {trace_path}")
    
    # Cycle to match duration
    if len(bandwidths) < duration_seconds:
        bandwidths = (bandwidths * (duration_seconds // len(bandwidths) + 1))[:duration_seconds]
    else:
        bandwidths = bandwidths[:duration_seconds]
    
    return bandwidths


def load_all_network_traces(base_dir: Path) -> Dict[str, List[float]]:
    """Load all available network traces"""
    traces = {}
    
    traces_dir = base_dir / "data" / "traces"
    
    # Load cellular traces
    cellular_dir = traces_dir / "cellular"
    if cellular_dir.exists():
        for trace_file in cellular_dir.glob("*.dat"):
            try:
                trace_name = trace_file.stem
                traces[trace_name] = load_network_trace(str(trace_file))
                print(f"  Loaded: {trace_name} (avg: {np.mean(traces[trace_name]):.2f} Mbps)")
            except Exception as e:
                print(f"  Warning: Could not load {trace_file.name}: {e}")
    
    # Load FCC traces (sample a few from train and test)
    fcc_base = Path("/Users/eunicechoi04/Downloads/6.5820/6.5820_pset_3-master/network/traces/fcc")
    if fcc_base.exists():
        # Sample 3 from train
        fcc_train = fcc_base / "train"
        if fcc_train.exists():
            train_files = sorted(fcc_train.glob("*.log"))[:3]
            for trace_file in train_files:
                try:
                    trace_name = f"fcc_train_{trace_file.stem[:15]}"
                    traces[trace_name] = load_network_trace(str(trace_file))
                    print(f"  Loaded: {trace_name} (avg: {np.mean(traces[trace_name]):.2f} Mbps)")
                except Exception as e:
                    print(f"  Warning: Could not load {trace_file.name}: {e}")
        
        # Sample 3 from test
        fcc_test = fcc_base / "test"
        if fcc_test.exists():
            test_files = sorted(fcc_test.glob("*.log"))[:3]
            for trace_file in test_files:
                try:
                    trace_name = f"fcc_test_{trace_file.stem[:15]}"
                    traces[trace_name] = load_network_trace(str(trace_file))
                    print(f"  Loaded: {trace_name} (avg: {np.mean(traces[trace_name]):.2f} Mbps)")
                except Exception as e:
                    print(f"  Warning: Could not load {trace_file.name}: {e}")
    
    return traces


def find_user_traces_for_video(video_id: str, data_dir: Path) -> List[Path]:
    """
    Find all user traces for a given video across all users.
    
    Args:
        video_id: Video identifier (e.g., "Diving-2OzlksZBTiA")
        data_dir: Base data directory (360_Video_analysis/data)
    
    Returns:
        List of paths to processed CSV files
    """
    user_traces = []
    
    # Iterate through all user directories
    for uid_dir in data_dir.glob("uid-*"):
        # Look in test directories
        for test_dir in uid_dir.glob("test*"):
            # Find matching video files
            for video_dir in test_dir.glob(f"*{video_id}*"):
                # Find processed CSV
                for csv_file in video_dir.glob("*_processed.csv"):
                    user_traces.append(csv_file)
    
    return user_traces


def get_tile_boundaries(frame_height: int, frame_width: int):
    """Calculate pixel boundaries for each tile in a 4x6 grid"""
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
    """Calculate motion intensity for each tile"""
    scores = []
    for (r_start, r_end, c_start, c_end) in tile_boundaries:
        tile_region = motion_frame[r_start:r_end, c_start:c_end]
        motion_score = np.mean(tile_region) / 255.0
        scores.append(motion_score)
    return np.array(scores)


def calculate_saliency_scores(saliency_frame, tile_boundaries):
    """Calculate average saliency for each tile"""
    scores = []
    for (r_start, r_end, c_start, c_end) in tile_boundaries:
        tile_region = saliency_frame[r_start:r_end, c_start:c_end]
        saliency_score = np.mean(tile_region) / 255.0
        scores.append(saliency_score)
    return np.array(scores)


def load_motion_saliency_scores(
    video_id: str,
    base_dir: Path,
    num_segments: int = 60
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load precomputed motion and saliency scores for a video.
    
    Extracts per-tile scores from motion and saliency videos.
    
    Returns:
        (motion_scores, saliency_scores) - each [num_segments, 24] or None
    """
    motion_dir = base_dir / "output" / "motion_maps"
    saliency_dir = base_dir / "output" / "saliency_videos_60s"
    
    motion_path = motion_dir / f"sparse_motion_{video_id}.mp4"
    saliency_path = saliency_dir / f"{video_id}_saliency_60s.mp4"
    
    if not motion_path.exists() or not saliency_path.exists():
        print(f"  Warning: Motion or saliency video not found for {video_id}")
        if not motion_path.exists():
            print(f"    Missing: {motion_path}")
        if not saliency_path.exists():
            print(f"    Missing: {saliency_path}")
        return None, None
    
    try:
        print(f"  Loading motion/saliency from videos...")
        
        # Open videos
        motion_cap = cv2.VideoCapture(str(motion_path))
        saliency_cap = cv2.VideoCapture(str(saliency_path))
        
        if not motion_cap.isOpened() or not saliency_cap.isOpened():
            print(f"  Warning: Could not open video files")
            return None, None
        
        # Get video properties
        frame_width = int(motion_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(motion_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(motion_cap.get(cv2.CAP_PROP_FPS))
        
        tile_boundaries = get_tile_boundaries(frame_height, frame_width)
        
        # Process frames and aggregate by second
        frames_per_segment = fps  # 1 second segments
        motion_scores_list = []
        saliency_scores_list = []
        
        frame_idx = 0
        current_motion_scores = []
        current_saliency_scores = []
        
        while len(motion_scores_list) < num_segments:
            motion_ret, motion_frame = motion_cap.read()
            saliency_ret, saliency_frame = saliency_cap.read()
            
            if not motion_ret or not saliency_ret:
                break
            
            # Convert to grayscale if needed
            if len(motion_frame.shape) == 3:
                motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
            if len(saliency_frame.shape) == 3:
                saliency_frame = cv2.cvtColor(saliency_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate scores for this frame
            motion_scores = calculate_motion_scores(motion_frame, tile_boundaries)
            saliency_scores = calculate_saliency_scores(saliency_frame, tile_boundaries)
            
            current_motion_scores.append(motion_scores)
            current_saliency_scores.append(saliency_scores)
            
            frame_idx += 1
            
            # Aggregate when we've seen enough frames for one segment
            if frame_idx % frames_per_segment == 0:
                # Average scores across frames in this segment
                avg_motion = np.mean(current_motion_scores, axis=0)
                avg_saliency = np.mean(current_saliency_scores, axis=0)
                
                motion_scores_list.append(avg_motion)
                saliency_scores_list.append(avg_saliency)
                
                current_motion_scores = []
                current_saliency_scores = []
        
        motion_cap.release()
        saliency_cap.release()
        
        if len(motion_scores_list) == 0:
            print(f"  Warning: No scores extracted from videos")
            return None, None
        
        motion_array = np.array(motion_scores_list)
        saliency_array = np.array(saliency_scores_list)
        
        print(f"  Loaded motion/saliency scores: shape {motion_array.shape}")
        
        return motion_array, saliency_array
        
    except Exception as e:
        print(f"  Error loading motion/saliency scores: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_experiment_wrapper(args):
    """Wrapper for parallel execution"""
    return run_single_experiment(*args)


def run_all_experiments(
    video_id: str,
    base_dir: Path,
    output_dir: Path,
    max_users: Optional[int] = None
):
    """
    Run experiments for one video across all users and network conditions.
    
    This is the main orchestration function, similar to run_exps.py main().
    """
    print(f"\n{'='*60}")
    print(f"Running Experiments for Video: {video_id}")
    print(f"{'='*60}\n")
    
    # 1. Load network traces
    print("[1] Loading Network Traces...")
    network_traces = load_all_network_traces(base_dir)
    print(f"  Total traces: {len(network_traces)}\n")
    
    # 2. Find user traces
    print("[2] Finding User Traces...")
    data_dir = base_dir / "360_Video_analysis" / "data"
    user_traces = find_user_traces_for_video(video_id, data_dir)
    
    if len(user_traces) == 0:
        print(f"  ERROR: No user traces found for {video_id}")
        return
    
    print(f"  Found {len(user_traces)} user traces")
    if max_users:
        user_traces = user_traces[:max_users]
        print(f"  Using first {max_users} users")
    
    for trace in user_traces[:3]:
        print(f"    - {trace.parent.parent.parent.name}/{trace.parent.name}")
    if len(user_traces) > 3:
        print(f"    ... and {len(user_traces) - 3} more")
    print()
    
    # 3. Load motion/saliency scores (if available)
    print("[3] Loading Motion/Saliency Scores...")
    motion_scores, saliency_scores = load_motion_saliency_scores(video_id, base_dir)
    print()
    
    # 4. Run experiments
    print("[4] Running Experiments...")
    print(f"  Total experiments: {len(user_traces)} users × {len(network_traces)} traces × 2 approaches")
    print(f"  = {len(user_traces) * len(network_traces) * 2} experiments\n")
    
    all_results = []
    
    # Prepare experiment arguments
    experiment_args = []
    
    for user_trace_path in user_traces:
        for trace_name, trace_data in network_traces.items():
            for approach in ['baseline', 'predictive_with_content']:
                experiment_args.append((
                    str(user_trace_path),
                    trace_data,
                    motion_scores,
                    saliency_scores,
                    approach
                ))
    
    # Run in parallel (or sequentially for debugging)
    use_parallel = True
    
    if use_parallel:
        print("  Running in parallel...")
        pool = ThreadPool(processes=8)
        results = pool.map(run_experiment_wrapper, experiment_args)
        pool.close()
        pool.join()
    else:
        print("  Running sequentially (for debugging)...")
        results = []
        for i, args in enumerate(experiment_args):
            print(f"    Experiment {i+1}/{len(experiment_args)}")
            result = run_experiment_wrapper(args)
            results.append(result)
    
    # Add network trace name to results
    result_idx = 0
    for user_trace_path in user_traces:
        for trace_name, trace_data in network_traces.items():
            for approach in ['baseline', 'predictive_with_content']:
                results[result_idx]['network_trace'] = trace_name
                results[result_idx]['video_id'] = video_id
                result_idx += 1
    
    # 5. Save results
    print("\n[5] Saving Results...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'segment_logs'}
        for r in results
    ])
    
    # Save CSV
    output_file = output_dir / f"{video_id}_experiments.csv"
    df_results.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    
    # Save detailed logs for a few experiments
    sample_results = results[:4]  # First 4 experiments
    for i, result in enumerate(sample_results):
        log_file = output_dir / f"{video_id}_exp{i}_detailed_log.json"
        with open(log_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    print(f"  Saved detailed logs for {len(sample_results)} experiments")
    
    # 6. Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Group by approach
    summary = df_results.groupby('approach').agg({
        'avg_qoe_per_segment': 'mean',
        'total_rebuf_time': 'mean',
        'rebuf_ratio': 'mean',
        'viewport_hit_rate': 'mean',
        'bandwidth_savings': 'mean',
        'total_mb_downloaded': 'mean'
    })
    
    print("\nAverage Performance by Approach:")
    print(summary)
    
    # Compare approaches
    baseline = df_results[df_results['approach'] == 'baseline']
    predictive = df_results[df_results['approach'] == 'predictive_with_content']
    
    print("\n" + "-"*60)
    print("KEY FINDINGS:")
    print("-"*60)
    print(f"Bandwidth Savings: {predictive['bandwidth_savings'].mean():.1%}")
    print(f"Viewport Hit Rate: {predictive['viewport_hit_rate'].mean():.1%}")
    print(f"QoE Improvement: {predictive['avg_qoe_per_segment'].mean() - baseline['avg_qoe_per_segment'].mean():.2f}")
    print(f"Rebuffering (Baseline): {baseline['total_rebuf_time'].mean():.2f}s")
    print(f"Rebuffering (Predictive): {predictive['total_rebuf_time'].mean():.2f}s")
    print(f"Rebuffer Change: {predictive['total_rebuf_time'].mean() - baseline['total_rebuf_time'].mean():.2f}s")
    
    # Per-network-trace analysis
    print("\n" + "-"*60)
    print("Performance by Network Trace:")
    print("-"*60)
    
    for trace_name in network_traces.keys():
        trace_results = df_results[df_results['network_trace'] == trace_name]
        baseline_trace = trace_results[trace_results['approach'] == 'baseline']
        predictive_trace = trace_results[trace_results['approach'] == 'predictive_with_content']
        
        if len(baseline_trace) > 0 and len(predictive_trace) > 0:
            print(f"\n{trace_name}:")
            print(f"  Bandwidth Savings: {predictive_trace['bandwidth_savings'].mean():.1%}")
            print(f"  Hit Rate: {predictive_trace['viewport_hit_rate'].mean():.1%}")
            print(f"  QoE Change: {predictive_trace['avg_qoe_per_segment'].mean() - baseline_trace['avg_qoe_per_segment'].mean():.2f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")
    
    return df_results


def main():
    """Main entry point"""
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "experiment_results"
    
    print("="*60)
    print("TILE-BASED ABR EXPERIMENT RUNNER")
    print("="*60)
    
    # Get video IDs from data/videos directory
    videos_dir = base_dir / "data" / "videos"
    video_ids = set()
    
    if videos_dir.exists():
        # Extract video IDs from filenames (remove extensions)
        for video_file in videos_dir.glob("*"):
            if video_file.suffix in [".mkv", ".webm", ".mp4"]:
                video_id = video_file.stem
                video_ids.add(video_id)
    
    # Also check data directory as fallback
    data_dir = base_dir / "360_Video_analysis" / "data"
    if data_dir.exists():
        for uid_dir in data_dir.glob("uid-*"):
            for test_dir in uid_dir.glob("test*"):
                for video_dir in test_dir.iterdir():
                    if video_dir.is_dir():
                        # Extract video ID from directory name (remove prefix)
                        dir_name = video_dir.name
                        # Format is typically: VideoName-VideoID or just VideoID
                        parts = dir_name.split('-')
                        if len(parts) >= 2:
                            video_id = parts[-1]  # Last part is usually the ID
                            video_ids.add(video_id)
    
    video_ids = sorted(video_ids)
    
    print(f"\nFound {len(video_ids)} videos:")
    for vid in video_ids[:5]:
        print(f"  - {vid}")
    if len(video_ids) > 5:
        print(f"  ... and {len(video_ids) - 5} more")
    
    # Run experiments for first video (or specify your video)
    if len(video_ids) > 0:
        for video_id in video_ids:
            # You can change this to any video ID you want to test
            test_video = video_id
            
            print(f"\nRunning experiments for: {test_video}")
            print("(To test other videos, modify the test_video variable in main())\n")
            
            run_all_experiments(
                video_id=test_video,
                base_dir=base_dir,
                output_dir=output_dir / test_video,
                max_users=5  # Limit to 5 users for faster testing
            )
    else:
        print("\nERROR: No videos found in data directory")


if __name__ == "__main__":
    main()
