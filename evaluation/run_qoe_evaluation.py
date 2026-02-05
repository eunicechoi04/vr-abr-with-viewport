"""
Run QoE Evaluation for Predictive Tiling

This script evaluates your tile prediction approach against a baseline
using real network traces and ground truth user viewing data.

Usage:
    python src/run_qoe_evaluation.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from final_tile_picking.tile_prediction_qoe_evaluator import (
    run_comparison_evaluation,
    integrate_with_content_aware_simulator
)


def load_network_trace(trace_path: str, duration_seconds: int = 60):
    """
    Load network trace from file

    Supports:
    - .dat files (Kbps, one value per line)
    - .mahi files (Mbps, one value per line)
    - .log files (Mbps, one value per line)
    """
    trace_path = Path(trace_path)

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    with open(trace_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse bandwidth values
    bandwidths = []
    for line in lines:
        try:
            bw = float(line)
            # Convert .dat files from Kbps to Mbps
            if trace_path.suffix == '.dat':
                bw = bw / 1000.0
            bandwidths.append(bw)
        except ValueError:
            continue

    if len(bandwidths) == 0:
        raise ValueError(f"No valid bandwidth values in trace: {trace_path}")

    # Truncate or cycle to match duration
    if len(bandwidths) < duration_seconds:
        # Cycle the trace
        bandwidths = (bandwidths * (duration_seconds // len(bandwidths) + 1))[:duration_seconds]
    else:
        bandwidths = bandwidths[:duration_seconds]

    print(f"Loaded trace: {trace_path.name}")
    print(f"  Samples: {len(bandwidths)}")
    print(f"  Avg BW: {np.mean(bandwidths):.2f} Mbps")
    print(f"  Min/Max: {np.min(bandwidths):.2f} / {np.max(bandwidths):.2f} Mbps")

    return bandwidths


def main():
    """
    Main evaluation pipeline
    """

    # Base directory
    base_dir = Path(__file__).parent.parent

    print("="*60)
    print("QoE EVALUATION FOR PREDICTIVE TILING")
    print("="*60)

    # ==========================================
    # STEP 1: Define network traces to test
    # ==========================================
    print("\n[1] Loading Network Traces...")

    network_traces = {}

    # Load cellular traces
    cellular_dir = base_dir / "data" / "traces" / "cellular"
    if cellular_dir.exists():
        for trace_file in ["ATT1.dat", "TMobile1.dat", "Verizon1.dat"]:
            trace_path = cellular_dir / trace_file
            if trace_path.exists():
                try:
                    trace_name = trace_file.replace(".dat", "")
                    network_traces[trace_name] = load_network_trace(str(trace_path))
                except Exception as e:
                    print(f"  Warning: Could not load {trace_file}: {e}")

    # Load simple traces
    for trace_file in ["bw3.mahi", "bw48.mahi"]:
        trace_path = base_dir / "data" / "traces" / trace_file
        if trace_path.exists():
            try:
                trace_name = trace_file.replace(".mahi", "")
                network_traces[trace_name] = load_network_trace(str(trace_path))
            except Exception as e:
                print(f"  Warning: Could not load {trace_file}: {e}")

    # Add synthetic traces for comparison
    network_traces['high_10mbps'] = [10.0] * 60
    network_traces['medium_5mbps'] = [5.0] * 60
    network_traces['low_2mbps'] = [2.0] * 60

    print(f"\n  Total traces loaded: {len(network_traces)}")

    # ==========================================
    # STEP 2: Process videos with motion/saliency
    # ==========================================
    print("\n[2] Finding Videos to Process...")

    motion_dir = base_dir / "output" / "motion_maps"
    saliency_dir = base_dir / "output" / "saliency_videos_60s"
    user_data_dir = base_dir / "360_Video_analysis" / "data"

    if not motion_dir.exists() or not saliency_dir.exists():
        print(f"ERROR: Motion/saliency directories not found!")
        print(f"  Motion dir: {motion_dir}")
        print(f"  Saliency dir: {saliency_dir}")
        return

    # Find all motion videos
    motion_videos = list(motion_dir.glob("sparse_motion_*.mp4"))

    if len(motion_videos) == 0:
        print(f"ERROR: No motion videos found in {motion_dir}")
        return

    print(f"  Found {len(motion_videos)} motion videos")

    # ==========================================
    # STEP 3: Run evaluation for each video
    # ==========================================
    print("\n[3] Running QoE Evaluation...")

    all_results = []

    for motion_path in motion_videos[:3]:  # Process first 3 videos as example
        video_id = motion_path.stem.replace("sparse_motion_", "")

        print(f"\n{'='*60}")
        print(f"Processing Video: {video_id}")
        print(f"{'='*60}")

        # Find corresponding saliency video
        saliency_path = saliency_dir / f"{video_id}_saliency_60s.mp4"
        if not saliency_path.exists():
            print(f"  WARNING: Saliency map not found, skipping...")
            continue

        # Find corresponding user trace (ground truth viewport)
        # Look for processed CSV files
        user_trace_file = None
        for uid_dir in user_data_dir.glob("uid-*"):
            for test_dir in uid_dir.glob("test0/*"):
                potential_file = test_dir / f"*{video_id}*_processed.csv"
                matches = list(test_dir.glob(f"*{video_id}*_processed.csv"))
                if matches:
                    user_trace_file = matches[0]
                    break
            if user_trace_file:
                break

        if not user_trace_file or not user_trace_file.exists():
            print(f"  WARNING: User trace not found for {video_id}, skipping...")
            continue

        print(f"  Motion: {motion_path.name}")
        print(f"  Saliency: {saliency_path.name}")
        print(f"  User trace: {user_trace_file.name}")

        # Run evaluation using content-aware predictions
        try:
            results = integrate_with_content_aware_simulator(
                motion_video_path=str(motion_path),
                saliency_video_path=str(saliency_path),
                user_trace_file=str(user_trace_file),
                network_traces=network_traces,
                output_dir=str(base_dir / "qoe_results")
            )
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ==========================================
    # STEP 4: Generate summary report
    # ==========================================
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)

        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)

        # Group by approach
        print("\nOverall Performance:")
        summary = combined.groupby('approach').agg({
            'qoe_score': 'mean',
            'rebuf_time': 'mean',
            'rebuf_ratio': 'mean',
            'avg_viewport_quality_mbps': 'mean',
            'bandwidth_savings': 'mean',
            'viewport_hit_rate': 'mean'
        })
        print(summary)

        # Save combined results
        output_file = base_dir / "qoe_results" / "combined_qoe_results.csv"
        combined.to_csv(output_file, index=False)
        print(f"\nCombined results saved to: {output_file}")

        # Calculate key metrics
        baseline = combined[combined['approach'] == 'baseline_full_quality']
        predictive = combined[combined['approach'] == 'predictive_tiling']

        if len(baseline) > 0 and len(predictive) > 0:
            print("\n" + "-"*60)
            print("KEY FINDINGS:")
            print("-"*60)
            print(f"Average Bandwidth Savings: {predictive['bandwidth_savings'].mean():.1%}")
            print(f"Average Viewport Hit Rate: {predictive['viewport_hit_rate'].mean():.1%}")
            print(f"QoE Improvement: {predictive['qoe_score'].mean() - baseline['qoe_score'].mean():.2f}")
            print(f"Rebuffering Time (Baseline): {baseline['rebuf_time'].mean():.2f}s")
            print(f"Rebuffering Time (Predictive): {predictive['rebuf_time'].mean():.2f}s")

    else:
        print("\nNo results generated. Check error messages above.")


if __name__ == "__main__":
    main()
