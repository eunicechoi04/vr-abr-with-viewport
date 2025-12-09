import os
import sys

# Add parent directory to path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.final_tile_picking.content_aware_abr_simulator import run_batch_simulation
from src.utils.trace_loader import load_all_traces_from_directory, get_summary_statistics, expand_values
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

# Paths
MOTION_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/motion_maps"
SALIENCY_DIR = "/Users/eunicechoi04/Downloads/videoabr/output/saliency_videos_60s"
TRACE_DIR = "/Users/eunicechoi04/Downloads/videoabr/data/traces"
OUTPUT_CSV = "/Users/eunicechoi04/Downloads/videoabr/output/content_aware_results.csv"
BATCHMAX = 5
# ==========================================
# RUN SIMULATION
# ==========================================
# Note: load_all_traces_from_directory is imported from trace_loader (line 8)
# It handles .mahi, .dat, and .log files from data/traces/

if __name__ == "__main__":
    print("=" * 60)
    print("Content-Aware ABR Simulator with Network Traces")
    print("=" * 60)
    print("\nNOTE: Simulating first 60 seconds at 30 FPS")
    print("=" * 60)

    # Load all traces from data/traces directory
    print(f"\nLoading traces from: {TRACE_DIR}")

    raw_traces = load_all_traces_from_directory(
        trace_dir=TRACE_DIR,
        sample_every_n=1  # No sampling - preserve original order
    )

    # Split each trace into 60-second batches
    # Each batch = 60 bandwidth values (one per second)
    BATCH_SIZE = 60
    network_traces = {}

    print(f"\nSplitting traces into {BATCH_SIZE}-second batches...")
    total_batches_created = 0
    print(list(raw_traces.items())[0])
    for trace_name, bandwidth_values in raw_traces.items():
        num_batches = min(BATCHMAX, len(bandwidth_values))
        if len(bandwidth_values) < BATCH_SIZE:
            network_traces[trace_name] = expand_values(bandwidth_values, BATCH_SIZE)
            continue
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_name = f"{trace_name}_batch{batch_idx}"
            network_traces[batch_name] = bandwidth_values[start_idx:end_idx]

        print(f"  {trace_name}: {num_batches} batches ({len(bandwidth_values)} values)")
        total_batches_created += num_batches

    print(f"\nTotal 60-second batches created: {total_batches_created}")
    print(f"Each batch will be run on all videos with matching motion+saliency maps")

    if len(network_traces) == 0:
        print("ERROR: No traces loaded! Check TRACE_DIR path.")
        sys.exit(1)

    # Print trace statistics
    print("\n" + "=" * 60)
    print("TRACE STATISTICS")
    print("=" * 60)
    stats = get_summary_statistics(network_traces)
    for trace_name, trace_stats in list(stats.items())[:5]:  # Show first 5
        print(f"\n{trace_name}:")
        print(f"  Samples: {trace_stats['samples']}")
        print(f"  Range:   {trace_stats['min_mbps']:.2f} - {trace_stats['max_mbps']:.2f} Mbps")
        print(f"  Mean:    {trace_stats['mean_mbps']:.2f} Mbps")

    if len(stats) > 5:
        print(f"\n... and {len(stats) - 5} more traces")

    print(f"\nTotal traces loaded: {len(network_traces)}")
    print("=" * 60)

    # Run the batch simulation
    print("\nStarting simulation...")
    results_df = run_batch_simulation(
        motion_dir=MOTION_DIR,
        saliency_dir=SALIENCY_DIR,
        network_traces=network_traces,
        output_csv=OUTPUT_CSV
    )

    # Analyze results
    if results_df is not None:
        print("\n" + "=" * 60)
        print("ADDITIONAL ANALYSIS")
        print("=" * 60)

        # Find best and worst network conditions
        avg_by_trace = results_df.groupby('network_trace').agg({
            'bandwidth_savings': 'mean',
            'stall_ratio': 'mean',
            'total_stall_time': 'mean'
        }).round(3)

        print("\nResults by Network Trace:")
        print(avg_by_trace)

        # Best conditions (highest savings, lowest stalls)
        best_trace = avg_by_trace['bandwidth_savings'].idxmax()
        worst_trace = avg_by_trace['stall_ratio'].idxmax()

        print(f"\nBest network condition: {best_trace}")
        print(f"  Bandwidth savings: {avg_by_trace.loc[best_trace, 'bandwidth_savings']:.2%}")
        print(f"  Stall ratio: {avg_by_trace.loc[best_trace, 'stall_ratio']:.2%}")

        print(f"\nWorst network condition: {worst_trace}")
        print(f"  Bandwidth savings: {avg_by_trace.loc[worst_trace, 'bandwidth_savings']:.2%}")
        print(f"  Stall ratio: {avg_by_trace.loc[worst_trace, 'stall_ratio']:.2%}")

        # Tradeoff analysis
        print("\n" + "=" * 60)
        print("BANDWIDTH-QUALITY TRADEOFF")
        print("=" * 60)
        print("\nTraces ranked by bandwidth savings:")
        sorted_savings = avg_by_trace.sort_values('bandwidth_savings', ascending=False)
        for trace_name, row in sorted_savings.iterrows():
            print(f"  {trace_name:20s}: {row['bandwidth_savings']:6.2%} savings, {row['stall_ratio']:6.2%} stall ratio")

        print("\nDone! Results saved to:", OUTPUT_CSV)
