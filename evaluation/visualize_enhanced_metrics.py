"""
Enhanced Visualization for Tile ABR Experiments

This script adds new visualizations for:
1. Rebuffering ratio and stall events
2. Weighted quality (proxy for PSNR/SSIM)
3. Unpredicted tile latency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_experiment_results(video_id: str, base_dir: Path) -> pd.DataFrame:
    """Load all experiment results for a video"""
    # Try different file naming patterns
    results_file = base_dir / video_id / f"{video_id}_experiments.csv"
    if not results_file.exists():
        results_file = base_dir / video_id / "results.csv"
    if not results_file.exists():
        # Try finding any CSV in the directory
        csv_files = list((base_dir / video_id).glob("*.csv"))
        if csv_files:
            results_file = csv_files[0]
        else:
            print(f"  ⚠ No results found for {video_id}")
            return None
    
    df = pd.read_csv(results_file)
    
    # Check for required enhanced metrics columns
    required_cols = ['stall_events', 'weighted_quality_mbps', 'wasted_bits_ratio', 
                     'user_data_coverage', 'total_unpredicted_latency']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠ Skipping {video_id} - missing enhanced metrics: {', '.join(missing_cols[:3])}")
        print(f"     Run experiments again to generate enhanced metrics")
        return None
    
    # Add network_type column if not present
    if 'network_type' not in df.columns and 'network_trace' in df.columns:
        def get_network_type(trace_name):
            trace_name = str(trace_name).lower()
            if 'att' in trace_name or 'tmobile' in trace_name or 'verizon' in trace_name:
                return 'cellular'
            elif 'fcc' in trace_name:
                if 'train' in trace_name:
                    return 'fcc_train'
                elif 'test' in trace_name:
                    return 'fcc_test'
                return 'fcc'
            else:
                return 'other'
        df['network_type'] = df['network_trace'].apply(get_network_type)
    
    # Add user_id column if not present
    if 'user_id' not in df.columns and 'user_trace_file' in df.columns:
        # Extract user ID from file path
        df['user_id'] = df['user_trace_file'].apply(lambda x: Path(x).parent.name if isinstance(x, str) else 'unknown')
    
    print(f"  Loaded {len(df)} experiment results from {results_file.name}")
    return df


def plot_rebuffering_analysis(df: pd.DataFrame, video_id: str, output_dir: Path):
    """
    Plot rebuffering ratio and stall events analysis
    Shows rebuffering severity and frequency
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{video_id}: Rebuffering Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rebuffering Ratio Distribution
    ax = axes[0, 0]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]['rebuf_ratio']
        ax.hist(data, alpha=0.6, label=approach, bins=20, edgecolor='black')
    ax.set_xlabel('Rebuffering Ratio (rebuf_time / video_duration)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Rebuffering Ratio Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Stall Events by Network
    ax = axes[0, 1]
    stall_by_network = df.groupby(['network_type', 'approach'])['stall_events'].mean().reset_index()
    pivot = stall_by_network.pivot(index='network_type', columns='approach', values='stall_events')
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_xlabel('Network Type', fontsize=11)
    ax.set_ylabel('Average Stall Events', fontsize=11)
    ax.set_title('Stall Events by Network Condition', fontweight='bold')
    ax.legend(title='Approach')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Stall Frequency (stalls per segment)
    ax = axes[1, 0]
    stall_freq_data = [df[df['approach'] == approach]['stall_frequency'] 
                       for approach in df['approach'].unique()]
    bp = ax.boxplot(stall_freq_data, labels=df['approach'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(df['approach'].unique()))):
        patch.set_facecolor(color)
    ax.set_xlabel('Approach', fontsize=11)
    ax.set_ylabel('Stall Frequency (stalls/segment)', fontsize=11)
    ax.set_title('Stall Frequency Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Rebuffering vs Bandwidth Savings
    ax = axes[1, 1]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]
        ax.scatter(data['bandwidth_savings'] * 100, data['rebuf_ratio'] * 100, 
                  label=approach, alpha=0.6, s=50)
    ax.set_xlabel('Bandwidth Savings (%)', fontsize=11)
    ax.set_ylabel('Rebuffering Ratio (%)', fontsize=11)
    ax.set_title('Bandwidth Savings vs Rebuffering', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{video_id}_rebuffering_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_quality_analysis(df: pd.DataFrame, video_id: str, output_dir: Path):
    """
    Plot weighted quality analysis (proxy for PSNR/SSIM)
    Higher values indicate better viewport quality
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{video_id}: Quality Analysis (Proxy for PSNR/SSIM)', fontsize=16, fontweight='bold')
    
    # 1. Weighted Quality Distribution
    ax = axes[0, 0]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]['weighted_quality_mbps']
        ax.hist(data, alpha=0.6, label=approach, bins=20, edgecolor='black')
    ax.set_xlabel('Weighted Quality (Mbps)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Weighted Viewport Quality Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(5.0, color='red', linestyle='--', alpha=0.5, label='High Quality (5 Mbps)')
    
    # 2. Quality CDF
    ax = axes[0, 1]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]['weighted_quality_mbps'].sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, label=approach, linewidth=2)
    ax.set_xlabel('Weighted Quality (Mbps)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Cumulative Distribution of Quality', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # 3. Quality vs Viewport Hit Rate
    ax = axes[1, 0]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]
        ax.scatter(data['viewport_hit_rate'] * 100, data['weighted_quality_mbps'], 
                  label=approach, alpha=0.6, s=50)
    ax.set_xlabel('Viewport Hit Rate (%)', fontsize=11)
    ax.set_ylabel('Weighted Quality (Mbps)', fontsize=11)
    ax.set_title('Quality vs Viewport Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Quality Improvement
    ax = axes[1, 1]
    if 'baseline' in df['approach'].values and 'predictive' in df['approach'].values:
        baseline_quality = df[df['approach'] == 'baseline']['weighted_quality_mbps']
        predictive_quality = df[df['approach'] == 'predictive']['weighted_quality_mbps']
        
        # Calculate improvement for matched pairs
        improvement = []
        for user in df['user_id'].unique():
            for net in df['network_trace'].unique():
                b = df[(df['approach'] == 'baseline') & (df['user_id'] == user) & (df['network_trace'] == net)]['weighted_quality_mbps']
                p = df[(df['approach'] == 'predictive') & (df['user_id'] == user) & (df['network_trace'] == net)]['weighted_quality_mbps']
                if len(b) > 0 and len(p) > 0:
                    improvement.append(p.values[0] - b.values[0])
        
        ax.hist(improvement, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax.axvline(np.mean(improvement), color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(improvement):.3f} Mbps')
        ax.set_xlabel('Quality Improvement (Mbps)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Predictive vs Baseline Quality Improvement', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{video_id}_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_wasted_bandwidth_analysis(df: pd.DataFrame, video_id: str, output_dir: Path):
    """
    Plot wasted bandwidth analysis
    Shows bits spent on tiles that were never viewed
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{video_id}: Wasted Bandwidth Analysis', fontsize=16, fontweight='bold')
    
    # 1. Wasted Bits Ratio Distribution
    ax = axes[0, 0]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]['wasted_bits_ratio'] * 100
        ax.hist(data, alpha=0.6, label=approach, bins=20, edgecolor='black')
    ax.set_xlabel('Wasted Bandwidth Ratio (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Percentage of Bits Never Viewed', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total Wasted Bits by Network
    ax = axes[0, 1]
    wasted_by_network = df.groupby(['network_type', 'approach'])['total_wasted_bits'].mean().reset_index()
    wasted_by_network['total_wasted_mb'] = wasted_by_network['total_wasted_bits'] / (8 * 1_000_000)
    pivot = wasted_by_network.pivot(index='network_type', columns='approach', values='total_wasted_mb')
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_xlabel('Network Type', fontsize=11)
    ax.set_ylabel('Wasted Bandwidth (MB)', fontsize=11)
    ax.set_title('Wasted Bandwidth by Network Condition', fontweight='bold')
    ax.legend(title='Approach')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. User Data Coverage
    ax = axes[1, 0]
    coverage_data = [df[df['approach'] == approach]['user_data_coverage'] * 100
                     for approach in df['approach'].unique()]
    bp = ax.boxplot(coverage_data, labels=df['approach'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(df['approach'].unique()))):
        patch.set_facecolor(color)
    ax.set_ylabel('User Data Coverage (%)', fontsize=11)
    ax.set_title('Segments with User Data', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='Full coverage')
    ax.legend()
    
    # 4. Wasted Bits vs Viewport Hit Rate
    ax = axes[1, 1]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]
        ax.scatter(data['viewport_hit_rate'] * 100, data['wasted_bits_ratio'] * 100,
                  label=approach, alpha=0.6, s=50)
    ax.set_xlabel('Viewport Hit Rate (%)', fontsize=11)
    ax.set_ylabel('Wasted Bandwidth Ratio (%)', fontsize=11)
    ax.set_title('Prediction Accuracy vs Wasted Bandwidth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{video_id}_wasted_bandwidth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency_analysis(df: pd.DataFrame, video_id: str, output_dir: Path):
    """
    Plot unpredicted tile latency analysis (parallel fetch model)
    Shows penalty when users look at tiles we didn't predict
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{video_id}: Unpredicted Tile Latency Analysis (Parallel Fetch)', fontsize=16, fontweight='bold')
    
    # 1. Latency Distribution
    ax = axes[0, 0]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]['avg_unpredicted_latency_per_segment'] * 1000  # to ms
        ax.hist(data, alpha=0.6, label=approach, bins=20, edgecolor='black')
    ax.set_xlabel('Avg Unpredicted Latency per Segment (ms)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Unpredicted Tile Latency Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total Latency by Network
    ax = axes[0, 1]
    latency_by_network = df.groupby(['network_type', 'approach'])['total_unpredicted_latency'].mean().reset_index()
    pivot = latency_by_network.pivot(index='network_type', columns='approach', values='total_unpredicted_latency')
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_xlabel('Network Type', fontsize=11)
    ax.set_ylabel('Total Unpredicted Latency (s)', fontsize=11)
    ax.set_title('Unpredicted Tile Latency by Network', fontweight='bold')
    ax.legend(title='Approach')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Latency vs Viewport Misses
    ax = axes[1, 0]
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]
        ax.scatter(data['viewport_misses'], data['total_unpredicted_latency'], 
                  label=approach, alpha=0.6, s=50)
    ax.set_xlabel('Viewport Misses', fontsize=11)
    ax.set_ylabel('Total Unpredicted Latency (s)', fontsize=11)
    ax.set_title('Latency vs Viewport Prediction Errors', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Latency Reduction
    ax = axes[1, 1]
    if 'baseline' in df['approach'].values and 'predictive' in df['approach'].values:
        baseline_latency = df[df['approach'] == 'baseline']['total_unpredicted_latency']
        predictive_latency = df[df['approach'] == 'predictive']['total_unpredicted_latency']
        
        ax.boxplot([baseline_latency, predictive_latency], 
                   labels=['Baseline', 'Predictive'],
                   patch_artist=True)
        ax.set_ylabel('Total Unpredicted Latency (s)', fontsize=11)
        ax.set_title('Latency Penalty Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean values
        means = [baseline_latency.mean(), predictive_latency.mean()]
        ax.plot([1, 2], means, 'ro-', linewidth=2, markersize=10, label='Mean')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{video_id}_latency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_qoe_cdfs_by_network(df: pd.DataFrame, video_id: str, output_dir: Path):
    """
    Plot QoE component CDFs comparing baseline vs predictive for WiFi and Cellular
    WiFi = FCC traces, Cellular = ATT/TMobile/Verizon
    """
    # Classify networks
    df_wifi = df[df['network_type'].str.contains('fcc', case=False, na=False)]
    df_cellular = df[~df['network_type'].str.contains('fcc', case=False, na=False)]
    
    # Create 2x2 subplot for WiFi and Cellular
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{video_id}: QoE Component CDFs - WiFi vs Cellular', fontsize=18, fontweight='bold')
    
    # Define QoE components to plot
    metrics = [
        ('weighted_quality_mbps', 'Quality (Mbps)'),
        ('rebuf_ratio', 'Rebuffering Ratio'),
        ('viewport_hit_rate', 'Viewport Hit Rate'),
        ('total_qoe', 'Total QoE')
    ]
    
    colors = {'baseline': '#e74c3c', 'predictive': '#3498db'}
    linestyles = {'WiFi': '-', 'Cellular': '--'}
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Plot WiFi networks
        for approach in ['baseline', 'predictive']:
            data_wifi = df_wifi[df_wifi['approach'] == approach][metric]
            if len(data_wifi) > 0:
                data_sorted = data_wifi.sort_values()
                cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                ax.plot(data_sorted, cdf, 
                       color=colors[approach], 
                       linestyle=linestyles['WiFi'],
                       linewidth=2.5, 
                       label=f'{approach.capitalize()} (WiFi)',
                       alpha=0.9)
        
        # Plot Cellular networks
        for approach in ['baseline', 'predictive']:
            data_cellular = df_cellular[df_cellular['approach'] == approach][metric]
            if len(data_cellular) > 0:
                data_sorted = data_cellular.sort_values()
                cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                ax.plot(data_sorted, cdf, 
                       color=colors[approach], 
                       linestyle=linestyles['Cellular'],
                       linewidth=2.5, 
                       label=f'{approach.capitalize()} (Cellular)',
                       alpha=0.9)
        
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('CDF', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        # Add median lines
        if metric in ['weighted_quality_mbps', 'viewport_hit_rate', 'total_qoe']:
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{video_id}_qoe_cdfs_wifi_cellular.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_metrics(df: pd.DataFrame, video_id: str, output_dir: Path):
    """
    Combined view of all new metrics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'{video_id}: Comprehensive Metrics Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Rebuffering Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]['rebuf_ratio'] * 100
        ax1.hist(data, alpha=0.6, label=approach, bins=15, edgecolor='black')
    ax1.set_xlabel('Rebuffering Ratio (%)', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Rebuffering', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Stall Events
    ax2 = fig.add_subplot(gs[0, 1])
    stall_data = [df[df['approach'] == approach]['stall_events'] 
                  for approach in df['approach'].unique()]
    bp = ax2.boxplot(stall_data, labels=df['approach'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(df['approach'].unique()))):
        patch.set_facecolor(color)
    ax2.set_ylabel('Stall Events', fontsize=10)
    ax2.set_title('Stall Frequency', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Weighted Quality
    ax3 = fig.add_subplot(gs[0, 2])
    quality_data = [df[df['approach'] == approach]['weighted_quality_mbps'] 
                    for approach in df['approach'].unique()]
    bp = ax3.boxplot(quality_data, labels=df['approach'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(df['approach'].unique()))):
        patch.set_facecolor(color)
    ax3.set_ylabel('Weighted Quality (Mbps)', fontsize=10)
    ax3.set_title('Viewport Quality', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(5.0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Unpredicted Latency
    ax4 = fig.add_subplot(gs[1, 0])
    latency_data = [df[df['approach'] == approach]['total_unpredicted_latency'] 
                    for approach in df['approach'].unique()]
    bp = ax4.boxplot(latency_data, labels=df['approach'].unique(), patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(df['approach'].unique()))):
        patch.set_facecolor(color)
    ax4.set_ylabel('Unpredicted Latency (s)', fontsize=10)
    ax4.set_title('Tile Fetch Latency', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. QoE vs Quality
    ax5 = fig.add_subplot(gs[1, 1])
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]
        ax5.scatter(data['weighted_quality_mbps'], data['total_qoe'], 
                   label=approach, alpha=0.6, s=30)
    ax5.set_xlabel('Weighted Quality (Mbps)', fontsize=10)
    ax5.set_ylabel('Total QoE', fontsize=10)
    ax5.set_title('QoE vs Quality', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Rebuffering vs Latency
    ax6 = fig.add_subplot(gs[1, 2])
    for approach in df['approach'].unique():
        data = df[df['approach'] == approach]
        ax6.scatter(data['total_rebuf_time'], data['total_unpredicted_latency'], 
                   label=approach, alpha=0.6, s=30)
    ax6.set_xlabel('Total Rebuffering (s)', fontsize=10)
    ax6.set_ylabel('Unpredicted Latency (s)', fontsize=10)
    ax6.set_title('Rebuffering vs Latency', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Performance Heatmap
    ax7 = fig.add_subplot(gs[2, :])
    
    # Calculate mean metrics by approach and network
    metrics = ['rebuf_ratio', 'stall_frequency', 'weighted_quality_mbps', 
               'viewport_hit_rate', 'wasted_bits_ratio', 'avg_unpredicted_latency_per_segment']
    
    heatmap_data = []
    labels = []
    for approach in df['approach'].unique():
        for net_type in sorted(df['network_type'].unique()):
            subset = df[(df['approach'] == approach) & (df['network_type'] == net_type)]
            if len(subset) > 0:
                row = [
                    subset['rebuf_ratio'].mean() * 100,
                    subset['stall_frequency'].mean() * 100,
                    subset['weighted_quality_mbps'].mean(),
                    subset['viewport_hit_rate'].mean() * 100,
                    subset['wasted_bits_ratio'].mean() * 100,
                    subset['avg_unpredicted_latency_per_segment'].mean() * 1000
                ]
                heatmap_data.append(row)
                labels.append(f'{approach}\n{net_type}')
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data).T
        im = ax7.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto', vmin=0)
        ax7.set_xticks(np.arange(len(labels)))
        ax7.set_yticks(np.arange(len(metrics)))
        ax7.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax7.set_yticklabels(['Rebuf %', 'Stall %', 'Quality Mbps', 'Hit %', 'Wasted %', 'Latency ms'], fontsize=9)
        ax7.set_title('Performance Heatmap by Approach & Network', fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
        cbar.set_label('Value', fontsize=10)
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(labels)):
                text = ax7.text(j, i, f'{heatmap_array[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=7)
    
    plt.savefig(output_dir / f'{video_id}_combined_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path("experiment_results")
    
    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return
    
    # Get all video directories
    video_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(video_dirs)} videos to process\n")
    
    for video_dir in sorted(video_dirs):
        video_id = video_dir.name
        print(f"Processing: {video_id}")
        
        # Load data
        df = load_experiment_results(video_id, base_dir)
        if df is None:
            continue
        
        # Create plots directory
        plots_dir = video_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"[1] Generating rebuffering analysis...")
        plot_rebuffering_analysis(df, video_id, plots_dir)
        print(f"  ✓ Saved: {video_id}_rebuffering_analysis.png")
        
        print(f"[2] Generating quality analysis...")
        plot_quality_analysis(df, video_id, plots_dir)
        print(f"  ✓ Saved: {video_id}_quality_analysis.png")
        
        print(f"[3] Generating wasted bandwidth analysis...")
        plot_wasted_bandwidth_analysis(df, video_id, plots_dir)
        print(f"  ✓ Saved: {video_id}_wasted_bandwidth_analysis.png")
        
        print(f"[4] Generating latency analysis...")
        plot_latency_analysis(df, video_id, plots_dir)
        print(f"  ✓ Saved: {video_id}_latency_analysis.png")
        
        print(f"[5] Generating QoE CDFs (WiFi vs Cellular)...")
        plot_qoe_cdfs_by_network(df, video_id, plots_dir)
        print(f"  ✓ Saved: {video_id}_qoe_cdfs_wifi_cellular.png")
        
        print(f"[6] Generating combined metrics dashboard...")
        plot_combined_metrics(df, video_id, plots_dir)
        print(f"  ✓ Saved: {video_id}_combined_metrics.png")
        
        print(f"✓ Complete! Generated 6 enhanced visualizations\n")
    
    print("\n" + "="*70)
    print("All visualizations complete!")
    print("="*70)


if __name__ == "__main__":
    main()
