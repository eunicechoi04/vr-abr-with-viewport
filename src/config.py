"""
Configuration file for 360° VR ABR streaming project.
Contains all constants and paths used throughout the project.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_ROOT = PROJECT_ROOT / "data" / "360dataset"
CONTENT_DIR = DATA_ROOT / "content"
SENSORY_DIR = DATA_ROOT / "sensory"

SALIENCY_DIR = CONTENT_DIR / "saliency"
MOTION_DIR = CONTENT_DIR / "motion"
ORIENTATION_DIR = SENSORY_DIR / "orientation"
RAW_SENSOR_DIR = SENSORY_DIR / "raw"
TILE_DIR = SENSORY_DIR / "tile"

# Output directories
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
TILES_OUTPUT = OUTPUT_ROOT / "tiles"
PREPROCESSED_OUTPUT = OUTPUT_ROOT / "preprocessed"
MODELS_OUTPUT = OUTPUT_ROOT / "models"
SIMULATIONS_OUTPUT = OUTPUT_ROOT / "simulations"
PLOTS_OUTPUT = OUTPUT_ROOT / "plots"

# =============================================================================
# DATASET CONSTANTS
# =============================================================================

# Video names in the dataset
VIDEO_NAMES = [
    "coaster",
    "coaster2",
    "diving",
    "drive",
    "game",
    "landscape",
    "pacman",
    "panel",
    "ride",
    "sport"
]

# Number of users per video
USERS_PER_VIDEO = 50

# Video properties
VIDEO_FPS = 30  # Frames per second
VIDEO_DURATION = 60  # Seconds (approximate)
TOTAL_FRAMES = VIDEO_FPS * VIDEO_DURATION  # 1800 frames

# =============================================================================
# TILE CONFIGURATION
# =============================================================================

# Tile grid for equirectangular projection (360° video)
TILE_ROWS = 10  # Latitude divisions
TILE_COLS = 20  # Longitude divisions
TOTAL_TILES = TILE_ROWS * TILE_COLS  # 200 tiles

# Tile size (pixels)
TILE_HEIGHT = 192
TILE_WIDTH = 192

# Equirectangular video dimensions (estimate)
# Typical 360° video: 3840x1920 (4K) or 1920x960
VIDEO_HEIGHT = TILE_HEIGHT * TILE_ROWS  # 1920
VIDEO_WIDTH = TILE_WIDTH * TILE_COLS  # 3840

# =============================================================================
# VIEWPORT CONFIGURATION
# =============================================================================

# Field of view (degrees)
FOV_HORIZONTAL = 110  # Typical VR headset FOV
FOV_VERTICAL = 90

# Viewport tile coverage (estimated)
# For 110° horizontal FOV and 20 columns: ~6-7 columns visible
# For 90° vertical FOV and 10 rows: ~2.5-3 rows visible
VIEWPORT_TILES_APPROX = 20  # Approximate tiles in viewport

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Sequence parameters
HISTORY_LENGTH = 30  # Frames (1 second at 30fps)
PREDICTION_HORIZONS = [30, 60, 90]  # 1s, 2s, 3s ahead

# Model hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Feature dimensions
SALIENCY_FEATURE_DIM = 128
MOTION_FEATURE_DIM = 128
ORIENTATION_DIM = 3  # yaw, pitch, roll
VELOCITY_DIM = 3  # angular velocities
VIDEO_EMBEDDING_DIM = 64

# LSTM parameters
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

# Attention parameters
ATTENTION_HEADS = 8

# =============================================================================
# TILE ENCODING CONFIGURATION
# =============================================================================

# Quality levels
QUALITY_LEVELS = {
    "high": {
        "crf": 18,  # Constant Rate Factor for H.264
        "bitrate_estimate_kb": 50,  # KB per tile
        "quality_score": 1.0
    },
    "medium": {
        "crf": 23,
        "bitrate_estimate_kb": 25,
        "quality_score": 0.7
    },
    "low": {
        "crf": 28,
        "bitrate_estimate_kb": 10,
        "quality_score": 0.3
    }
}

# Segment duration for ABR (seconds)
SEGMENT_DURATION = 2.0

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

# Bandwidth profiles (bits per second)
BANDWIDTH_PROFILES = {
    "stable": {
        "mean": 5_000_000,  # 5 Mbps
        "std": 0
    },
    "variable": {
        "mean": 5_000_000,  # 5 Mbps
        "std": 2_000_000,  # ±2 Mbps
        "min": 2_000_000,  # 2 Mbps
        "max": 10_000_000  # 10 Mbps
    },
    "poor": {
        "mean": 2_000_000,  # 2 Mbps
        "std": 500_000,  # ±0.5 Mbps
        "min": 1_000_000,  # 1 Mbps
        "max": 3_000_000  # 3 Mbps
    }
}

# Buffer parameters
INITIAL_BUFFER = 5.0  # seconds
MIN_BUFFER_THRESHOLD = 2.0  # seconds (start prioritizing any quality)
MAX_BUFFER_THRESHOLD = 10.0  # seconds (can be selective)

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Train/validation/test splits
TRAIN_RATIO = 0.8  # 40 users for training
VAL_RATIO = 0.1  # 5 users for validation
TEST_RATIO = 0.1  # 5 users for testing

# Metric goals
BANDWIDTH_SAVINGS_GOAL = 0.50  # 50% reduction
VIEWPORT_QUALITY_GOAL = 0.80  # Quality score >0.8
STALL_RATIO_GOAL = 0.02  # <2% stall time
PREDICTION_IOU_GOAL = 0.70  # >70% IoU at 1s horizon

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all necessary output directories if they don't exist."""
    directories = [
        OUTPUT_ROOT,
        TILES_OUTPUT,
        PREPROCESSED_OUTPUT,
        MODELS_OUTPUT,
        SIMULATIONS_OUTPUT,
        PLOTS_OUTPUT
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_orientation_file(video_name: str, user_id) -> Path:
    """Get path to orientation CSV file.

    Args:
        video_name: Name of the video
        user_id: User ID (int 1-50, or str '01'-'50')
    """
    user_str = f"{int(user_id):02d}" if isinstance(user_id, int) else user_id
    return ORIENTATION_DIR / f"{video_name}_user{user_str}_orientation.csv"

def get_tile_file(video_name: str, user_id) -> Path:
    """Get path to tile CSV file.

    Args:
        video_name: Name of the video
        user_id: User ID (int 1-50, or str '01'-'50')
    """
    user_str = f"{int(user_id):02d}" if isinstance(user_id, int) else user_id
    return TILE_DIR / f"{video_name}_user{user_str}_tile.csv"

def get_saliency_file(video_name: str) -> Path:
    """Get path to saliency video file."""
    return SALIENCY_DIR / f"{video_name}.mp4"

def get_motion_file(video_name: str) -> Path:
    """Get path to motion video file."""
    return MOTION_DIR / f"{video_name}.mp4"

def get_all_sessions():
    """Get list of all (video_name, user_id) pairs.

    Returns:
        List of (video_name, user_id) tuples where user_id is 1-50
    """
    sessions = []
    for video_name in VIDEO_NAMES:
        for user_id in range(1, USERS_PER_VIDEO + 1):  # 1 to 50
            sessions.append((video_name, user_id))
    return sessions


if __name__ == "__main__":
    # Test configuration
    print("=== 360° VR ABR Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"\nDataset:")
    print(f"  Videos: {len(VIDEO_NAMES)}")
    print(f"  Users per video: {USERS_PER_VIDEO}")
    print(f"  Total sessions: {len(VIDEO_NAMES) * USERS_PER_VIDEO}")
    print(f"\nTile Grid: {TILE_ROWS}x{TILE_COLS} = {TOTAL_TILES} tiles")
    print(f"Video Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
    print(f"\nPrediction Horizons: {PREDICTION_HORIZONS} frames")
    print(f"Quality Levels: {list(QUALITY_LEVELS.keys())}")

    # Ensure directories exist
    ensure_directories()
    print("\nOutput directories created successfully.")
