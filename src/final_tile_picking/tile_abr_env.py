"""
Tile-based ABR Environment - mimics 6.5820 pset3 structure

This is the equivalent of sim/env.py but for 360 tile streaming.
Handles segment-by-segment simulation with real network traces.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LinearRegression

# ==========================================
# CONFIGURATION
# ==========================================
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
SEGMENT_DURATION = 1.0  # seconds

# Bitrates
BITRATE_LOW_KBPS = 500
BITRATE_HIGH_KBPS = 5000

# Prediction
HISTORY_WINDOW = 0.5  # seconds to look back
PREDICTION_HORIZON = 1.0  # seconds to predict ahead

# Tile selection weights (from compute_with_viewport.py)
MOTION_WEIGHT = 0.005
SALIENCY_WEIGHT = 0.095
VIEWPORT_WEIGHT = 0.9

# QoE weights
QOE_QUALITY_WEIGHT = 1.0
QOE_REBUF_PENALTY = 4.3
QOE_VIEWPORT_MISS_PENALTY = 2.0

# Buffer
BUFFER_INIT = 0.0
BUFFER_MAX = 60.0

# Latency penalty for fetching unpredicted high-quality tiles
# Simulates delay to fetch tile from edge server when not pre-cached
UNPREDICTED_TILE_LATENCY_MS = 150  # milliseconds


class TileABREnv:
    """
    Environment for tile-based 360 video streaming simulation.
    
    Similar to 6.5820's Env class, but for tile selection instead of bitrate selection.
    """
    
    def __init__(
        self,
        user_trace_df: pd.DataFrame,
        network_trace_mbps: List[float],
        motion_scores: Optional[np.ndarray] = None,
        saliency_scores: Optional[np.ndarray] = None,
        num_high_quality_tiles: int = 4
    ):
        """
        Args:
            user_trace_df: DataFrame with columns [timestamp, latitude_rad, longitude_rad, tile_id]
            network_trace_mbps: List of bandwidth values (Mbps) per second
            motion_scores: Optional precomputed motion scores per tile per segment [num_segments, 24]
            saliency_scores: Optional precomputed saliency scores per tile per segment [num_segments, 24]
            num_high_quality_tiles: How many tiles to download at high quality
        """
        self.user_trace_df = user_trace_df
        self.network_trace = network_trace_mbps
        self.motion_scores = motion_scores
        self.saliency_scores = saliency_scores
        self.num_high_quality_tiles = num_high_quality_tiles
        
        # Calculate video duration
        self.video_duration = user_trace_df['timestamp'].max()
        self.num_segments = int(self.video_duration / SEGMENT_DURATION)
        
        # State
        self.buffer = BUFFER_INIT
        self.segment_idx = 0
        self.total_rebuf_time = 0.0
        self.total_qoe = 0.0
        
        # Metrics
        self.viewport_hits = 0
        self.viewport_misses = 0
        self.total_bits_downloaded = 0
        self.total_wasted_bits = 0  # Bits spent on tiles never viewed
        
        # New metrics for enhanced analysis
        self.stall_events = 0  # Count of rebuffering events
        self.total_weighted_quality = 0.0  # Quality weighted by viewport coverage
        self.total_unpredicted_latency = 0.0  # Latency from unpredicted tile lookups (parallel fetch model)
        self.segments_with_user_data = 0  # Count segments where user data exists for accuracy
        
        # Logs
        self.segment_logs = []
        
    def predict_next_tile(self, current_time: float) -> Optional[int]:
        """
        Predict the tile user will view in the next segment using linear regression
        on the last HISTORY_WINDOW seconds of head movement data.
        
        This mimics abr_simulator.py's approach.
        """
        # Get history window
        history_start = max(0, current_time - HISTORY_WINDOW)
        mask = (self.user_trace_df['timestamp'] >= history_start) & \
               (self.user_trace_df['timestamp'] <= current_time)
        history_df = self.user_trace_df[mask]
        
        if len(history_df) < 5:
            # Not enough data for prediction
            return None
        
        # Fit linear regression
        X = history_df['timestamp'].values.reshape(-1, 1)
        y_lat = history_df['latitude_rad'].values
        y_lon = history_df['longitude_rad'].values
        
        model_lat = LinearRegression().fit(X, y_lat)
        model_lon = LinearRegression().fit(X, y_lon)
        
        # Predict future position
        future_time = np.array([[current_time + PREDICTION_HORIZON]])
        pred_lat = model_lat.predict(future_time)[0]
        pred_lon = model_lon.predict(future_time)[0]
        
        # Convert to tile ID
        return self._lat_lon_to_tile(pred_lat, pred_lon)
    
    def _lat_lon_to_tile(self, lat: float, lon: float) -> int:
        """Convert latitude/longitude to tile ID"""
        # Latitude: -π/2 to π/2, map to rows 0-3
        row = int(np.floor((np.clip(lat, -1.57, 1.57) + 1.57) / (3.14 / TILE_ROWS)))
        row = max(0, min(row, TILE_ROWS - 1))
        
        # Longitude: wrap around, map to cols 0-5
        lon_normalized = ((lon + 3.14) % 6.28)
        col = int(np.floor(lon_normalized / (6.28 / TILE_COLS)))
        col = max(0, min(col, TILE_COLS - 1))
        
        return row * TILE_COLS + col
    
    def get_neighbor_tiles(self, center_tile: int) -> List[int]:
        """Get neighboring tiles (including center)"""
        row = center_tile // TILE_COLS
        col = center_tile % TILE_COLS
        
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_row = (row + dr) % TILE_ROWS  # Wrap vertically
                new_col = (col + dc) % TILE_COLS  # Wrap horizontally
                neighbors.append(new_row * TILE_COLS + new_col)
        
        return list(set(neighbors))  # Remove duplicates
    
    def select_tiles_with_content(
        self,
        predicted_tile: Optional[int],
        segment_idx: int
    ) -> List[int]:
        """
        Select tiles combining:
        1. Prediction from user head movement (viewport score)
        2. Motion scores (from precomputed motion maps)
        3. Saliency scores (from precomputed saliency maps)
        
        Uses weighted sum approach from compute_with_viewport.py
        
        Returns list of tile IDs to download at high quality.
        """
        # Initialize scores for all tiles
        viewport_scores = np.zeros(TOTAL_TILES)
        motion_scores = np.zeros(TOTAL_TILES)
        saliency_scores = np.zeros(TOTAL_TILES)
        
        # 1. Viewport scores (from prediction)
        if predicted_tile is not None:
            # Center tile gets full score
            viewport_scores[predicted_tile] = 1.0
            
            # Neighboring tiles get 0.5
            neighbors = self.get_neighbor_tiles(predicted_tile)
            for neighbor_idx in neighbors:
                if neighbor_idx != predicted_tile:
                    viewport_scores[neighbor_idx] = 0.5
        else:
            # No prediction - use uniform scores (fallback)
            viewport_scores = np.ones(TOTAL_TILES) * 0.1
        
        # 2. Motion scores (from precomputed)
        if self.motion_scores is not None and segment_idx < len(self.motion_scores):
            motion_scores = self.motion_scores[segment_idx]
        
        # 3. Saliency scores (from precomputed)
        if self.saliency_scores is not None and segment_idx < len(self.saliency_scores):
            saliency_scores = self.saliency_scores[segment_idx]
        
        # Normalize each score type to [0, 1]
        if np.max(motion_scores) > 0:
            norm_motion = motion_scores / np.max(motion_scores)
        else:
            norm_motion = motion_scores
        
        if np.max(saliency_scores) > 0:
            norm_saliency = saliency_scores / np.max(saliency_scores)
        else:
            norm_saliency = saliency_scores
        
        norm_viewport = viewport_scores  # Already 0-1
        
        # Weighted sum (from compute_with_viewport.py)
        combined_scores = (
            MOTION_WEIGHT * norm_motion +
            SALIENCY_WEIGHT * norm_saliency +
            VIEWPORT_WEIGHT * norm_viewport
        )
        
        # Select top N tiles
        top_tile_indices = np.argsort(combined_scores)[-self.num_high_quality_tiles:]
        
        # Return reversed list (highest score first)
        return top_tile_indices.tolist()[::-1]
    
    def step(self, high_quality_tiles: List[int]) -> Dict:
        """
        Simulate downloading one segment.
        
        This is equivalent to env.step() in 6.5820.
        
        Args:
            high_quality_tiles: List of tile IDs to download at high quality
            
        Returns:
            Dictionary with step results
        """
        # Get bandwidth for this segment
        bandwidth_mbps = self.network_trace[self.segment_idx % len(self.network_trace)]
        
        # Calculate segment size
        num_high = len(high_quality_tiles)
        num_low = TOTAL_TILES - num_high
        
        segment_size_bits = (
            num_high * BITRATE_HIGH_KBPS * 1000 * SEGMENT_DURATION +
            num_low * BITRATE_LOW_KBPS * 1000 * SEGMENT_DURATION
        )
        
        segment_size_mb = segment_size_bits / (8 * 1_000_000)
        self.total_bits_downloaded += segment_size_bits
        
        # Simulate download
        download_time = segment_size_mb / max(bandwidth_mbps, 0.01)
        
        # Calculate rebuffering
        rebuf_time = 0.0
        stall_occurred = False
        if download_time > self.buffer:
            rebuf_time = download_time - self.buffer
            self.buffer = 0.0
            stall_occurred = True
            self.stall_events += 1
        else:
            self.buffer -= download_time
        
        # Add segment duration to buffer
        self.buffer = min(self.buffer + SEGMENT_DURATION, BUFFER_MAX)
        
        self.total_rebuf_time += rebuf_time
        
        # Evaluate prediction accuracy
        current_time = self.segment_idx * SEGMENT_DURATION
        future_time = current_time + PREDICTION_HORIZON
        
        # Get actual tiles user viewed in the next segment
        # Check if user data exists for this time window with proper alignment
        # User data is at 100 Hz (10ms intervals), video is 30 fps (~33ms frames)
        # Ensure we have sufficient samples in the prediction window
        mask = (self.user_trace_df['timestamp'] >= future_time) & \
               (self.user_trace_df['timestamp'] < future_time + SEGMENT_DURATION)
        
        user_data_exists = mask.sum() > 0  # Check if any user data in this segment
        
        if user_data_exists:
            actual_tiles_viewed = self.user_trace_df[mask]['tile_id'].unique()
            self.segments_with_user_data += 1
        else:
            # No user data for this segment - skip accuracy measurement
            actual_tiles_viewed = np.array([])
        
        # Calculate viewport quality and wasted bandwidth
        tiles_hit = [t for t in actual_tiles_viewed if t in high_quality_tiles]
        tiles_missed = [t for t in actual_tiles_viewed if t not in high_quality_tiles]
        
        # Calculate wasted bits: high-quality tiles that were never viewed
        tiles_not_viewed = [t for t in high_quality_tiles if t not in actual_tiles_viewed]
        wasted_bits = len(tiles_not_viewed) * (BITRATE_HIGH_KBPS - BITRATE_LOW_KBPS) * 1000 * SEGMENT_DURATION
        self.total_wasted_bits += wasted_bits
        
        # Only update accuracy metrics if user data exists
        if user_data_exists:
            self.viewport_hits += len(tiles_hit)
            self.viewport_misses += len(tiles_missed)
        
        # Calculate latency penalty for unpredicted tiles
        # Using PARALLEL fetch model: max RTT, not sum (HTTP/2 multiplexing)
        # If any tiles are missed, we incur one RTT penalty to fetch them in parallel
        if len(tiles_missed) > 0:
            unpredicted_latency = UNPREDICTED_TILE_LATENCY_MS / 1000.0  # Single RTT
        else:
            unpredicted_latency = 0.0
        self.total_unpredicted_latency += unpredicted_latency
        
        # Calculate viewport quality (proxy for PSNR/SSIM)
        # Only calculate if user data exists for proper alignment
        num_viewport_tiles = len(actual_tiles_viewed)
        if user_data_exists and num_viewport_tiles > 0:
            viewport_quality_kbps = (
                len(tiles_hit) * BITRATE_HIGH_KBPS +
                len(tiles_missed) * BITRATE_LOW_KBPS
            ) / num_viewport_tiles
            
            # Weighted quality: weight by time spent viewing each tile
            # User data at 100 Hz, video at 30 fps - align by counting samples per tile
            mask_df = self.user_trace_df[mask]
            tile_viewing_time = mask_df.groupby('tile_id').size()
            total_samples = len(mask_df)
            
            weighted_quality = 0.0
            for tile_id in actual_tiles_viewed:
                tile_weight = tile_viewing_time.get(tile_id, 0) / max(total_samples, 1)
                tile_quality = BITRATE_HIGH_KBPS if tile_id in high_quality_tiles else BITRATE_LOW_KBPS
                weighted_quality += tile_weight * tile_quality
            
            self.total_weighted_quality += weighted_quality
        else:
            # No user data - use default quality (no viewport hits/misses)
            viewport_quality_kbps = BITRATE_LOW_KBPS  # Conservative: assume low quality
            weighted_quality = viewport_quality_kbps
        
        segment_qoe = (
            QOE_QUALITY_WEIGHT * (viewport_quality_kbps / 1000.0) -
            QOE_REBUF_PENALTY * rebuf_time -
            QOE_VIEWPORT_MISS_PENALTY * len(tiles_missed) -
            0.1 * unpredicted_latency  # Small penalty for unpredicted tile latency
        )
        
        self.total_qoe += segment_qoe
        
        # Log segment
        log_entry = {
            'segment': self.segment_idx,
            'bandwidth_mbps': bandwidth_mbps,
            'buffer_level': self.buffer,
            'rebuf_time': rebuf_time,
            'stall_occurred': stall_occurred,
            'download_time': download_time,
            'high_quality_tiles': high_quality_tiles,
            'actual_tiles_viewed': list(actual_tiles_viewed),
            'tiles_hit': tiles_hit,
            'tiles_missed': tiles_missed,
            'tiles_not_viewed': tiles_not_viewed,
            'wasted_bits': wasted_bits,
            'user_data_exists': user_data_exists,
            'segment_qoe': segment_qoe,
            'viewport_quality_kbps': viewport_quality_kbps,
            'weighted_quality_kbps': weighted_quality,
            'unpredicted_latency': unpredicted_latency
        }
        self.segment_logs.append(log_entry)
        
        self.segment_idx += 1
        
        return log_entry
    
    def get_results(self) -> Dict:
        """Get final simulation results"""
        baseline_bits = self.num_segments * TOTAL_TILES * BITRATE_HIGH_KBPS * 1000 * SEGMENT_DURATION
        
        return {
            'num_segments': self.num_segments,
            'video_duration': self.video_duration,
            
            # QoE
            'total_qoe': self.total_qoe,
            'avg_qoe_per_segment': self.total_qoe / max(self.num_segments, 1),
            
            # Rebuffering
            'total_rebuf_time': self.total_rebuf_time,
            'rebuf_ratio': self.total_rebuf_time / self.video_duration,
            'stall_events': self.stall_events,
            'stall_frequency': self.stall_events / max(self.num_segments, 1),  # stalls per segment
            
            # Quality metrics (proxy for PSNR/SSIM)
            'avg_weighted_quality_kbps': self.total_weighted_quality / max(self.segments_with_user_data, 1),
            'weighted_quality_mbps': (self.total_weighted_quality / max(self.segments_with_user_data, 1)) / 1000.0,
            
            # Viewport accuracy (only from segments with user data)
            'viewport_hits': self.viewport_hits,
            'viewport_misses': self.viewport_misses,
            'viewport_hit_rate': self.viewport_hits / max(self.viewport_hits + self.viewport_misses, 1),
            'segments_with_user_data': self.segments_with_user_data,
            'user_data_coverage': self.segments_with_user_data / max(self.num_segments, 1),
            
            # Wasted bandwidth
            'total_wasted_bits': self.total_wasted_bits,
            'wasted_bits_ratio': self.total_wasted_bits / max(self.total_bits_downloaded, 1),
            
            # Latency penalties
            'total_unpredicted_latency': self.total_unpredicted_latency,
            'avg_unpredicted_latency_per_segment': self.total_unpredicted_latency / max(self.num_segments, 1),
            
            # Bandwidth
            'total_bits_downloaded': self.total_bits_downloaded,
            'total_mb_downloaded': self.total_bits_downloaded / (8 * 1_000_000),
            'bandwidth_savings': 1.0 - (self.total_bits_downloaded / baseline_bits),
            'avg_bandwidth_mbps': (self.total_bits_downloaded / (8 * 1_000_000)) / self.video_duration,
            
            # Logs
            'segment_logs': self.segment_logs
        }


def run_single_experiment(
    user_trace_path: str,
    network_trace_mbps: List[float],
    motion_scores: Optional[np.ndarray] = None,
    saliency_scores: Optional[np.ndarray] = None,
    approach: str = 'predictive_with_content'
) -> Dict:
    """
    Run a single experiment - equivalent to sim/run_exp.py
    
    Args:
        user_trace_path: Path to user head movement CSV
        network_trace_mbps: Bandwidth trace
        motion_scores: Precomputed motion scores [num_segments, 24]
        saliency_scores: Precomputed saliency scores [num_segments, 24]
        approach: 'baseline' or 'predictive_with_content'
    """
    # Load user trace
    user_df = pd.read_csv(user_trace_path)
    
    # Create environment
    env = TileABREnv(
        user_trace_df=user_df,
        network_trace_mbps=network_trace_mbps,
        motion_scores=motion_scores,
        saliency_scores=saliency_scores
    )
    
    # Run simulation
    for seg_idx in range(env.num_segments):
        current_time = seg_idx * SEGMENT_DURATION
        
        if approach == 'baseline':
            # Baseline: download all tiles at high quality
            high_quality_tiles = list(range(TOTAL_TILES))
        else:
            # Predictive approach
            predicted_tile = env.predict_next_tile(current_time)
            high_quality_tiles = env.select_tiles_with_content(predicted_tile, seg_idx)
        
        # Step
        env.step(high_quality_tiles)
    
    # Get results
    results = env.get_results()
    results['approach'] = approach
    results['user_trace_file'] = Path(user_trace_path).name
    
    return results
