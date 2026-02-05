"""
Visualize QoE Evaluation Results

Creates plots comparing baseline vs predictive tiling approaches
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def plot_qoe_comparison(results_csv: str, output_dir: str = "qoe_results/plots"):
    """
    Create comprehensive QoE comparison plots

    Args:
        results_csv: Path to combined_qoe_results.csv
        output_dir: Where to save plots
    """

    # Load results
    df = pd.read_csv(results_csv)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Split by approach
    baseline = df[df['approach'] == 'baseline_full_quality']
    predictive = df[df['approach'] == 'predictive_tiling']

    if len(baseline) == 0 or len(predictive) == 0:
        print("Error: Need both baseline and predictive results")
        return

    # ==========================================
    # Plot 1: QoE Score Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(baseline['network_trace'].unique()))
    width = 0.35

    traces = baseline['network_trace'].unique()
    baseline_scores = [baseline[baseline['network_trace'] == t]['qoe_score'].mean() for t in traces]
    predictive_scores = [predictive[predictive['network_trace'] == t]['qoe_score'].mean() for t in traces]

    ax.bar(x - width/2, baseline_scores, width, label='Baseline (Full Quality)', alpha=0.8)
    ax.bar(x + width/2, predictive_scores, width, label='Predictive Tiling', alpha=0.8)

    ax.set_xlabel('Network Condition', fontsize=12)
    ax.set_ylabel('QoE Score', fontsize=12)
    ax.set_title('QoE Score Comparison: Baseline vs Predictive Tiling', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(traces, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/qoe_score_comparison.png", dpi=300)
    print(f"Saved: {output_dir}/qoe_score_comparison.png")

    # ==========================================
    # Plot 2: Rebuffering Time Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_rebuf = [baseline[baseline['network_trace'] == t]['rebuf_time'].mean() for t in traces]
    predictive_rebuf = [predictive[predictive['network_trace'] == t]['rebuf_time'].mean() for t in traces]

    ax.bar(x - width/2, baseline_rebuf, width, label='Baseline', alpha=0.8, color='green')
    ax.bar(x + width/2, predictive_rebuf, width, label='Predictive', alpha=0.8, color='orange')

    ax.set_xlabel('Network Condition', fontsize=12)
    ax.set_ylabel('Rebuffering Time (seconds)', fontsize=12)
    ax.set_title('Rebuffering Time Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(traces, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rebuffering_comparison.png", dpi=300)
    print(f"Saved: {output_dir}/rebuffering_comparison.png")

    # ==========================================
    # Plot 3: Bandwidth Savings
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    savings = [predictive[predictive['network_trace'] == t]['bandwidth_savings'].mean() * 100
              for t in traces]

    ax.bar(traces, savings, alpha=0.8, color='steelblue')
    ax.axhline(y=50, color='r', linestyle='--', label='Target: 50% savings')

    ax.set_xlabel('Network Condition', fontsize=12)
    ax.set_ylabel('Bandwidth Savings (%)', fontsize=12)
    ax.set_title('Bandwidth Savings with Predictive Tiling', fontsize=14)
    ax.set_xticklabels(traces, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(savings):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bandwidth_savings.png", dpi=300)
    print(f"Saved: {output_dir}/bandwidth_savings.png")

    # ==========================================
    # Plot 4: Viewport Hit Rate
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    hit_rates = [predictive[predictive['network_trace'] == t]['viewport_hit_rate'].mean() * 100
                for t in traces]

    ax.bar(traces, hit_rates, alpha=0.8, color='coral')
    ax.axhline(y=80, color='g', linestyle='--', label='Target: 80% accuracy')

    ax.set_xlabel('Network Condition', fontsize=12)
    ax.set_ylabel('Viewport Hit Rate (%)', fontsize=12)
    ax.set_title('Tile Prediction Accuracy', fontsize=14)
    ax.set_xticklabels(traces, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    for i, v in enumerate(hit_rates):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/viewport_hit_rate.png", dpi=300)
    print(f"Saved: {output_dir}/viewport_hit_rate.png")

    # ==========================================
    # Plot 5: Quality vs Bandwidth Trade-off
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot: bandwidth usage vs viewport quality
    ax.scatter(baseline['total_mb_downloaded'], baseline['avg_viewport_quality_mbps'],
              s=100, alpha=0.6, label='Baseline', marker='o')
    ax.scatter(predictive['total_mb_downloaded'], predictive['avg_viewport_quality_mbps'],
              s=100, alpha=0.6, label='Predictive', marker='^')

    ax.set_xlabel('Total Bandwidth Used (MB)', fontsize=12)
    ax.set_ylabel('Average Viewport Quality (Mbps)', fontsize=12)
    ax.set_title('Quality vs Bandwidth Trade-off', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/quality_bandwidth_tradeoff.png", dpi=300)
    print(f"Saved: {output_dir}/quality_bandwidth_tradeoff.png")

    # ==========================================
    # Plot 6: Summary Dashboard
    # ==========================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Overall metrics
    metrics = ['QoE Score', 'Rebuf Time (s)', 'Viewport Quality (Mbps)', 'Hit Rate (%)']
    baseline_vals = [
        baseline['qoe_score'].mean(),
        baseline['rebuf_time'].mean(),
        baseline['avg_viewport_quality_mbps'].mean(),
        100.0  # Baseline always has 100% hit rate
    ]
    predictive_vals = [
        predictive['qoe_score'].mean(),
        predictive['rebuf_time'].mean(),
        predictive['avg_viewport_quality_mbps'].mean(),
        predictive['viewport_hit_rate'].mean() * 100
    ]

    # QoE Score
    ax1.bar(['Baseline', 'Predictive'], [baseline_vals[0], predictive_vals[0]], alpha=0.8)
    ax1.set_ylabel('QoE Score')
    ax1.set_title('Overall QoE Score')
    ax1.grid(axis='y', alpha=0.3)

    # Rebuffering
    ax2.bar(['Baseline', 'Predictive'], [baseline_vals[1], predictive_vals[1]], alpha=0.8, color='orange')
    ax2.set_ylabel('Seconds')
    ax2.set_title('Average Rebuffering Time')
    ax2.grid(axis='y', alpha=0.3)

    # Viewport Quality
    ax3.bar(['Baseline', 'Predictive'], [baseline_vals[2], predictive_vals[2]], alpha=0.8, color='green')
    ax3.set_ylabel('Mbps')
    ax3.set_title('Average Viewport Quality')
    ax3.grid(axis='y', alpha=0.3)

    # Hit Rate
    ax4.bar(['Baseline', 'Predictive'], [baseline_vals[3], predictive_vals[3]], alpha=0.8, color='purple')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Viewport Hit Rate')
    ax4.set_ylim([0, 105])
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('QoE Evaluation Summary: Baseline vs Predictive Tiling', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_dashboard.png", dpi=300)
    print(f"Saved: {output_dir}/summary_dashboard.png")

    # ==========================================
    # Print Summary Statistics
    # ==========================================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print("\nBaseline (Full Quality):")
    print(f"  Average QoE Score: {baseline['qoe_score'].mean():.2f}")
    print(f"  Average Rebuffering: {baseline['rebuf_time'].mean():.2f}s")
    print(f"  Average Bandwidth: {baseline['total_mb_downloaded'].mean():.2f} MB")

    print("\nPredictive Tiling:")
    print(f"  Average QoE Score: {predictive['qoe_score'].mean():.2f}")
    print(f"  Average Rebuffering: {predictive['rebuf_time'].mean():.2f}s")
    print(f"  Average Bandwidth: {predictive['total_mb_downloaded'].mean():.2f} MB")
    print(f"  Average Hit Rate: {predictive['viewport_hit_rate'].mean()*100:.1f}%")
    print(f"  Average Savings: {predictive['bandwidth_savings'].mean()*100:.1f}%")

    print("\nDifferences:")
    qoe_diff = predictive['qoe_score'].mean() - baseline['qoe_score'].mean()
    rebuf_diff = predictive['rebuf_time'].mean() - baseline['rebuf_time'].mean()
    print(f"  QoE Change: {qoe_diff:+.2f} ({qoe_diff/baseline['qoe_score'].mean()*100:+.1f}%)")
    print(f"  Rebuffering Change: {rebuf_diff:+.2f}s")
    print(f"  Bandwidth Reduction: {predictive['bandwidth_savings'].mean()*100:.1f}%")

    plt.close('all')


def main():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "qoe_results" / "combined_qoe_results.csv"

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run `python src/run_qoe_evaluation.py` first to generate results.")
        sys.exit(1)

    print(f"Loading results from: {results_file}")
    plot_qoe_comparison(str(results_file), str(base_dir / "qoe_results" / "plots"))

    print("\n✅ Visualization complete! Check qoe_results/plots/ for PNG files.")


if __name__ == "__main__":
    main()
