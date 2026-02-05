"""
Visualize Tile-based ABR Experiment Results

This script creates comprehensive visualizations of the experiment results,
including CDF plots (like the FCC graph shown) and additional insightful analyses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def compute_cdf(data):
    """Compute CDF values for data"""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


def plot_cdf_comparison(df, metric, title, xlabel, output_path, approaches=None):
    """
    Create CDF plots comparing different approaches and network conditions.
    Similar to the FCC graph style.
    """
    if approaches is None:
        approaches = df['approach'].unique()
    
    # Get unique network traces
    network_traces = df['network_trace'].unique()
    
    # Select interesting network traces (up to 3 for clarity)
    if len(network_traces) > 3:
        # Prioritize FCC traces
        fcc_traces = [t for t in network_traces if 'fcc' in t.lower()]
        cellular_traces = [t for t in network_traces if t not in fcc_traces]
        selected_traces = fcc_traces[:2] + cellular_traces[:1] if fcc_traces else network_traces[:3]
    else:
        selected_traces = network_traces
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':']
    
    color_idx = 0
    
    for trace in selected_traces:
        for approach in approaches:
            # Filter data
            mask = (df['network_trace'] == trace) & (df['approach'] == approach)
            data = df[mask][metric].values
            
            if len(data) == 0:
                continue
            
            # Compute CDF
            sorted_data, cdf = compute_cdf(data)
            
            # Compute average
            avg = np.mean(data)
            
            # Create label
            trace_short = trace[:10] if len(trace) > 10 else trace
            approach_short = 'bb' if approach == 'baseline' else 'pred'
            label = f"{trace_short}-{approach_short}(Avg: {avg:.2f})"
            
            # Plot
            linestyle = '-' if approach == 'baseline' else '--'
            ax.plot(sorted_data, cdf, label=label, linewidth=2.5, 
                   color=colors[color_idx % len(colors)], linestyle=linestyle)
            
            color_idx += 1
    
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel('CDF', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_4panel_cdf(df, output_path):
    """
    Create 4-panel CDF plot similar to the reference image.
    Shows: quality score, rebuf penalty, smooth penalty, net qoe
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get baseline and predictive data
    baseline = df[df['approach'] == 'baseline']
    predictive = df[df['approach'] == 'predictive_with_content']
    
    # Select network traces for comparison
    network_traces = df['network_trace'].unique()
    fcc_traces = [t for t in network_traces if 'fcc' in t.lower()]
    selected_traces = fcc_traces[:3] if len(fcc_traces) >= 3 else network_traces[:3]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Panel 1: Quality Score (viewport quality / bits downloaded)
    ax = axes[0, 0]
    for idx, trace in enumerate(selected_traces):
        for approach, ls in [('baseline', '-'), ('predictive_with_content', '--')]:
            mask = (df['network_trace'] == trace) & (df['approach'] == approach)
            data = df[mask]
            
            if len(data) == 0:
                continue
            
            # Calculate quality score (viewport hits as proxy)
            quality_score = data['viewport_hit_rate'].values * 100
            sorted_data, cdf = compute_cdf(quality_score)
            
            avg = np.mean(quality_score)
            trace_short = trace.split('_')[-1][:10] if '_' in trace else trace[:10]
            approach_short = 'bb' if approach == 'baseline' else 'pred'
            label = f"{trace_short}-{approach_short}(Avg: {avg:.2f})"
            
            ax.plot(sorted_data, cdf, label=label, linewidth=2.5, 
                   color=colors[idx], linestyle=ls)
    
    ax.set_xlabel('quality score', fontsize=12, fontweight='bold')
    ax.set_ylabel('CDF', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Panel 2: Rebuffer Penalty (negative rebuffer time)
    ax = axes[0, 1]
    for idx, trace in enumerate(selected_traces):
        for approach, ls in [('baseline', '-'), ('predictive_with_content', '--')]:
            mask = (df['network_trace'] == trace) & (df['approach'] == approach)
            data = df[mask]
            
            if len(data) == 0:
                continue
            
            rebuf_penalty = -data['total_rebuf_time'].values
            sorted_data, cdf = compute_cdf(rebuf_penalty)
            
            avg = np.mean(rebuf_penalty)
            trace_short = trace.split('_')[-1][:10] if '_' in trace else trace[:10]
            approach_short = 'bb' if approach == 'baseline' else 'pred'
            label = f"{trace_short}-{approach_short}(Avg: {avg:.2f})"
            
            ax.plot(sorted_data, cdf, label=label, linewidth=2.5, 
                   color=colors[idx], linestyle=ls)
    
    ax.set_xlabel('rebuf penalty', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Panel 3: Smooth Penalty (bandwidth variation)
    ax = axes[1, 0]
    for idx, trace in enumerate(selected_traces):
        for approach, ls in [('baseline', '-'), ('predictive_with_content', '--')]:
            mask = (df['network_trace'] == trace) & (df['approach'] == approach)
            data = df[mask]
            
            if len(data) == 0:
                continue
            
            # Use negative viewport misses as smooth penalty proxy
            smooth_penalty = -data['viewport_misses'].values.astype(float) / 10.0
            sorted_data, cdf = compute_cdf(smooth_penalty)
            
            avg = np.mean(smooth_penalty)
            trace_short = trace.split('_')[-1][:10] if '_' in trace else trace[:10]
            approach_short = 'bb' if approach == 'baseline' else 'pred'
            label = f"{trace_short}-{approach_short}(Avg: {avg:.2f})"
            
            ax.plot(sorted_data, cdf, label=label, linewidth=2.5, 
                   color=colors[idx], linestyle=ls)
    
    ax.set_xlabel('smooth penalty', fontsize=12, fontweight='bold')
    ax.set_ylabel('CDF', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Panel 4: Net QoE
    ax = axes[1, 1]
    for idx, trace in enumerate(selected_traces):
        for approach, ls in [('baseline', '-'), ('predictive_with_content', '--')]:
            mask = (df['network_trace'] == trace) & (df['approach'] == approach)
            data = df[mask]
            
            if len(data) == 0:
                continue
            
            net_qoe = data['avg_qoe_per_segment'].values
            sorted_data, cdf = compute_cdf(net_qoe)
            
            avg = np.mean(net_qoe)
            trace_short = trace.split('_')[-1][:10] if '_' in trace else trace[:10]
            approach_short = 'bb' if approach == 'baseline' else 'pred'
            label = f"{trace_short}-{approach_short}(Avg: {avg:.2f})"
            
            ax.plot(sorted_data, cdf, label=label, linewidth=2.5, 
                   color=colors[idx], linestyle=ls)
    
    ax.set_xlabel('net qoe', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle('QoE Component CDFs: Baseline vs Predictive Tiling', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_bandwidth_savings_by_network(df, output_path):
    """Show bandwidth savings across different network conditions"""
    predictive = df[df['approach'] == 'predictive_with_content']
    
    # Group by network trace
    savings_by_network = predictive.groupby('network_trace').agg({
        'bandwidth_savings': ['mean', 'std', 'count']
    }).reset_index()
    
    savings_by_network.columns = ['network_trace', 'mean', 'std', 'count']
    savings_by_network = savings_by_network.sort_values('mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(savings_by_network))
    bars = ax.bar(x, savings_by_network['mean'] * 100, 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(x, savings_by_network['mean'] * 100, 
                yerr=savings_by_network['std'] * 100,
                fmt='none', ecolor='darkred', capsize=5, capthick=2, alpha=0.7)
    
    # Color bars by savings level
    for i, (bar, val) in enumerate(zip(bars, savings_by_network['mean'])):
        if val > 0.7:
            bar.set_color('green')
            bar.set_alpha(0.7)
        elif val > 0.5:
            bar.set_color('orange')
            bar.set_alpha(0.7)
    
    ax.set_xlabel('Network Trace', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bandwidth Savings (%)', fontsize=13, fontweight='bold')
    ax.set_title('Bandwidth Savings by Network Condition\n(Predictive vs Baseline)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(savings_by_network['network_trace'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=75, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target: 75%')
    ax.legend()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, savings_by_network['mean'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_qoe_vs_hit_rate(df, output_path):
    """Scatter plot showing relationship between viewport hit rate and QoE"""
    predictive = df[df['approach'] == 'predictive_with_content']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Group by network trace for coloring
    network_traces = predictive['network_trace'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(network_traces)))
    
    for idx, trace in enumerate(network_traces):
        data = predictive[predictive['network_trace'] == trace]
        
        ax.scatter(data['viewport_hit_rate'] * 100, 
                  data['avg_qoe_per_segment'],
                  label=trace[:15], s=100, alpha=0.6, 
                  color=colors[idx], edgecolors='black', linewidth=1)
    
    # Add trend line
    x = predictive['viewport_hit_rate'].values * 100
    y = predictive['avg_qoe_per_segment'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label=f'Trend (slope={z[0]:.2f})')
    
    ax.set_xlabel('Viewport Hit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average QoE per Segment', fontsize=13, fontweight='bold')
    ax.set_title('QoE vs Viewport Prediction Accuracy\n(Predictive Tiling)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_rebuffering_comparison(df, output_path):
    """Box plot comparing rebuffering times"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    baseline = df[df['approach'] == 'baseline']
    predictive = df[df['approach'] == 'predictive_with_content']
    
    data_to_plot = [
        baseline['total_rebuf_time'].values,
        predictive['total_rebuf_time'].values
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['Baseline', 'Predictive Tiling'],
                    patch_artist=True, widths=0.6, showfliers=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    # Color boxes
    colors = ['salmon', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Total Rebuffering Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Rebuffering Time Distribution\nBaseline vs Predictive Tiling', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    means = [np.mean(data) for data in data_to_plot]
    ax.plot([1, 2], means, 'D', markersize=10, color='purple', 
            label=f'Means: {means[0]:.1f}s vs {means[1]:.1f}s', zorder=5)
    ax.legend(fontsize=11)
    
    # Add improvement percentage
    improvement = (1 - means[1] / means[0]) * 100
    ax.text(1.5, max(means) * 1.1, f'Improvement: {improvement:.1f}%', 
            ha='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_network_performance_heatmap(df, output_path):
    """Heatmap showing performance across users and network conditions"""
    predictive = df[df['approach'] == 'predictive_with_content']
    
    # Create pivot table: users vs network traces
    pivot_data = predictive.pivot_table(
        values='avg_qoe_per_segment',
        index='user_trace_file',
        columns='network_trace',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=pivot_data.values.mean(), cbar_kws={'label': 'Avg QoE'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('QoE Heatmap: Users vs Network Conditions\n(Predictive Tiling)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Network Trace', fontsize=12, fontweight='bold')
    ax.set_ylabel('User Trace', fontsize=12, fontweight='bold')
    
    # Shorten labels
    ax.set_yticklabels([label.get_text()[:30] for label in ax.get_yticklabels()], 
                       rotation=0, fontsize=9)
    ax.set_xticklabels([label.get_text()[:15] for label in ax.get_xticklabels()], 
                       rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_viewport_accuracy_distribution(df, output_path):
    """Distribution of viewport prediction accuracy"""
    predictive = df[df['approach'] == 'predictive_with_content']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    hit_rates = predictive['viewport_hit_rate'].values * 100
    ax1.hist(hit_rates, bins=20, color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axvline(hit_rates.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {hit_rates.mean():.1f}%')
    ax1.axvline(np.median(hit_rates), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(hit_rates):.1f}%')
    ax1.set_xlabel('Viewport Hit Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Viewport Prediction Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # CDF
    sorted_rates, cdf = compute_cdf(hit_rates)
    ax2.plot(sorted_rates, cdf, linewidth=3, color='darkblue')
    ax2.fill_between(sorted_rates, 0, cdf, alpha=0.3, color='skyblue')
    ax2.axvline(hit_rates.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Viewport Hit Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CDF', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution of Hit Rate', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_efficiency_frontier(df, output_path):
    """Plot showing trade-off between bandwidth savings and QoE"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    baseline = df[df['approach'] == 'baseline']
    predictive = df[df['approach'] == 'predictive_with_content']
    
    # Plot baseline points
    ax.scatter(baseline['bandwidth_savings'] * 100, 
              baseline['avg_qoe_per_segment'],
              s=100, alpha=0.3, color='red', marker='s', 
              label='Baseline', edgecolors='darkred', linewidth=1)
    
    # Plot predictive points colored by network
    network_traces = predictive['network_trace'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(network_traces)))
    
    for idx, trace in enumerate(network_traces):
        data = predictive[predictive['network_trace'] == trace]
        ax.scatter(data['bandwidth_savings'] * 100,
                  data['avg_qoe_per_segment'],
                  s=150, alpha=0.7, color=colors[idx],
                  label=f'Pred-{trace[:10]}', edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Bandwidth Savings (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average QoE per Segment', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency Frontier: Bandwidth vs QoE\n(Higher Right is Better)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=predictive['avg_qoe_per_segment'].median(), 
               color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def plot_summary_metrics(df, output_path):
    """Create summary comparison table visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Calculate summary statistics
    baseline = df[df['approach'] == 'baseline']
    predictive = df[df['approach'] == 'predictive_with_content']
    
    metrics = {
        'Metric': [
            'Avg QoE per Segment',
            'Total Rebuffer Time (s)',
            'Rebuffer Ratio',
            'Viewport Hit Rate (%)',
            'Bandwidth Savings (%)',
            'Total MB Downloaded',
            'Avg Bandwidth (Mbps)'
        ],
        'Baseline': [
            f"{baseline['avg_qoe_per_segment'].mean():.2f}",
            f"{baseline['total_rebuf_time'].mean():.2f}",
            f"{baseline['rebuf_ratio'].mean():.2f}",
            f"{baseline['viewport_hit_rate'].mean() * 100:.2f}",
            f"{baseline['bandwidth_savings'].mean() * 100:.2f}",
            f"{baseline['total_mb_downloaded'].mean():.2f}",
            f"{baseline['avg_bandwidth_mbps'].mean():.2f}"
        ],
        'Predictive': [
            f"{predictive['avg_qoe_per_segment'].mean():.2f}",
            f"{predictive['total_rebuf_time'].mean():.2f}",
            f"{predictive['rebuf_ratio'].mean():.2f}",
            f"{predictive['viewport_hit_rate'].mean() * 100:.2f}",
            f"{predictive['bandwidth_savings'].mean() * 100:.2f}",
            f"{predictive['total_mb_downloaded'].mean():.2f}",
            f"{predictive['avg_bandwidth_mbps'].mean():.2f}"
        ],
        'Improvement': []
    }
    
    # Calculate improvements
    for i in range(len(metrics['Metric'])):
        base_val = float(metrics['Baseline'][i])
        pred_val = float(metrics['Predictive'][i])
        
        if 'Rebuffer' in metrics['Metric'][i] or 'Downloaded' in metrics['Metric'][i]:
            # Lower is better
            improvement = (1 - pred_val / base_val) * 100 if base_val != 0 else 0
        else:
            # Higher is better
            improvement = ((pred_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
        
        metrics['Improvement'].append(f"{improvement:+.1f}%")
    
    # Create table
    df_table = pd.DataFrame(metrics)
    
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                    cellLoc='center', loc='center', 
                    colColours=['lightgray']*4,
                    cellColours=[['white']*4]*len(df_table))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color improvement column
    for i in range(1, len(df_table) + 1):
        cell = table[(i, 3)]
        improvement_val = float(metrics['Improvement'][i-1].replace('%', '').replace('+', ''))
        if improvement_val > 0:
            cell.set_facecolor('lightgreen')
        elif improvement_val < 0:
            cell.set_facecolor('lightcoral')
        cell.set_text_props(weight='bold')
    
    # Bold headers
    for i in range(4):
        table[(0, i)].set_text_props(weight='bold', size=12)
    
    ax.set_title('Performance Summary: Baseline vs Predictive Tiling', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()


def main():
    """Main execution"""
    print("="*70)
    print("TILE-BASED ABR EXPERIMENT VISUALIZATION")
    print("="*70)
    
    # Find experiment results
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "experiment_results"
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    # Find all experiment CSV files
    csv_files = list(results_dir.glob("*/*_experiments.csv"))
    
    if len(csv_files) == 0:
        print(f"ERROR: No experiment CSV files found in {results_dir}")
        return
    
    print(f"\nFound {len(csv_files)} experiment result file(s)")
    
    # Process each experiment
    for csv_file in csv_files:
        video_id = csv_file.parent.name
        print(f"\n{'='*70}")
        print(f"Processing: {video_id}")
        print(f"{'='*70}")
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} experiment results")
        
        # Create output directory for plots
        plots_dir = csv_file.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        print(f"  Output directory: {plots_dir}")
        
        print(f"\n[1] Generating CDF plots...")
        # 4-panel CDF (like reference image)
        plot_4panel_cdf(df, plots_dir / f"{video_id}_4panel_cdf.png")
        
        print(f"\n[2] Generating performance comparisons...")
        # Bandwidth savings
        plot_bandwidth_savings_by_network(df, plots_dir / f"{video_id}_bandwidth_savings.png")
        
        # Rebuffering comparison
        plot_rebuffering_comparison(df, plots_dir / f"{video_id}_rebuffering_comparison.png")
        
        print(f"\n[3] Generating accuracy analyses...")
        # Viewport accuracy
        plot_viewport_accuracy_distribution(df, plots_dir / f"{video_id}_viewport_accuracy.png")
        
        # QoE vs hit rate
        plot_qoe_vs_hit_rate(df, plots_dir / f"{video_id}_qoe_vs_hitrate.png")
        
        print(f"\n[4] Generating advanced visualizations...")
        # Heatmap
        plot_network_performance_heatmap(df, plots_dir / f"{video_id}_performance_heatmap.png")
        
        # Efficiency frontier
        plot_efficiency_frontier(df, plots_dir / f"{video_id}_efficiency_frontier.png")
        
        # Summary table
        plot_summary_metrics(df, plots_dir / f"{video_id}_summary_table.png")
        
        print(f"\n✓ Complete! Generated 8 visualizations in {plots_dir}")
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nPlots saved to: {results_dir}")


if __name__ == "__main__":
    main()
