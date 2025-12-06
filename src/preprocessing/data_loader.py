"""
Data loader for 360° VR dataset.
Handles loading orientation data, tile data, saliency videos, and motion videos.
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    VIDEO_NAMES,
    USERS_PER_VIDEO,
    get_orientation_file,
    get_tile_file,
    get_saliency_file,
    get_motion_file,
    get_all_sessions
)


class DataLoader:
    """
    Loads and processes data from the 360° VR dataset.
    """

    def __init__(self):
        """Initialize the DataLoader."""
        self.video_names = VIDEO_NAMES
        self.users_per_video = USERS_PER_VIDEO

    def load_orientation_data(self, video_name: str, user_id: int) -> pd.DataFrame:
        """
        Load orientation data for a specific user and video.

        Args:
            video_name: Name of the video (e.g., 'coaster', 'diving')
            user_id: User ID (1-50)

        Returns:
            DataFrame with columns: frame_no, raw_x, raw_y, raw_z,
                                   raw_yaw, raw_pitch, raw_roll,
                                   cal_yaw, cal_pitch, cal_roll
        """
        filepath = get_orientation_file(video_name, user_id)

        if not filepath.exists():
            raise FileNotFoundError(f"Orientation file not found: {filepath}")

        # Read CSV
        df = pd.read_csv(filepath, skipinitialspace=True)

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Rename columns for easier access
        column_mapping = {
            'no. frames': 'frame_no',
            'raw x': 'raw_x',
            'raw y': 'raw_y',
            'raw z': 'raw_z',
            'raw yaw': 'raw_yaw',
            'raw pitch': 'raw_pitch',
            'raw roll': 'raw_roll',
            'cal. yaw': 'cal_yaw',
            'cal. pitch': 'cal_pitch',
            'cal. roll': 'cal_roll'
        }
        df = df.rename(columns=column_mapping)

        # Convert frame_no to integer
        df['frame_no'] = df['frame_no'].astype(int)

        return df

    def load_tile_data(self, video_name: str, user_id: int) -> pd.DataFrame:
        """
        Load tile data for a specific user and video.

        Args:
            video_name: Name of the video
            user_id: User ID (1-50)

        Returns:
            DataFrame with columns: frame_no, tile_numbers (list of integers)
        """
        filepath = get_tile_file(video_name, user_id)

        if not filepath.exists():
            raise FileNotFoundError(f"Tile file not found: {filepath}")

        # Read CSV manually since tile numbers are comma-separated in a single column
        data = []
        with open(filepath, 'r') as f:
            # Skip header
            header = f.readline()

            for line in f:
                parts = line.strip().split(', ')
                if len(parts) < 2:
                    continue

                frame_no = int(parts[0])
                tile_numbers = [int(x) for x in parts[1:]]
                data.append({
                    'frame_no': frame_no,
                    'tile_numbers': tile_numbers
                })

        df = pd.DataFrame(data)
        return df

    def load_saliency_video(self, video_name: str) -> np.ndarray:
        """
        Load saliency map video.

        Args:
            video_name: Name of the video

        Returns:
            numpy array of shape [T, H, W, C] where T is number of frames
        """
        filepath = get_saliency_file(video_name)

        if not filepath.exists():
            raise FileNotFoundError(f"Saliency video not found: {filepath}")

        return self._load_video(filepath)

    def load_motion_video(self, video_name: str) -> np.ndarray:
        """
        Load optical flow motion video.

        Args:
            video_name: Name of the video

        Returns:
            numpy array of shape [T, H, W, C] where T is number of frames
        """
        filepath = get_motion_file(video_name)

        if not filepath.exists():
            raise FileNotFoundError(f"Motion video not found: {filepath}")

        return self._load_video(filepath)

    def _load_video(self, filepath: Path) -> np.ndarray:
        """
        Load video file using OpenCV.

        Args:
            filepath: Path to video file

        Returns:
            numpy array of shape [T, H, W, C]
        """
        cap = cv2.VideoCapture(str(filepath))

        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {filepath}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {filepath}")

        return np.array(frames)

    def get_all_sessions(self) -> List[Tuple[str, int]]:
        """
        Get list of all available (video_name, user_id) pairs.

        Returns:
            List of tuples (video_name, user_id)
        """
        return get_all_sessions()

    def get_session_info(self, video_name: str, user_id: int) -> dict:
        """
        Get summary information about a session.

        Args:
            video_name: Name of the video
            user_id: User ID (1-50)

        Returns:
            Dictionary with session information
        """
        orientation_df = self.load_orientation_data(video_name, user_id)
        tile_df = self.load_tile_data(video_name, user_id)

        num_frames = len(orientation_df)
        avg_tiles_per_frame = tile_df['tile_numbers'].apply(len).mean()

        # Calculate angular statistics
        yaw_range = (orientation_df['cal_yaw'].min(), orientation_df['cal_yaw'].max())
        pitch_range = (orientation_df['cal_pitch'].min(), orientation_df['cal_pitch'].max())

        return {
            'video_name': video_name,
            'user_id': user_id,
            'num_frames': num_frames,
            'avg_tiles_per_frame': avg_tiles_per_frame,
            'yaw_range': yaw_range,
            'pitch_range': pitch_range
        }


def load_session_data(video_name: str, user_id: int) -> dict:
    """
    Convenience function to load all data for a session.

    Args:
        video_name: Name of the video
        user_id: User ID (1-50)

    Returns:
        Dictionary with orientation_df, tile_df, and metadata
    """
    loader = DataLoader()

    orientation_df = loader.load_orientation_data(video_name, user_id)
    tile_df = loader.load_tile_data(video_name, user_id)

    return {
        'video_name': video_name,
        'user_id': user_id,
        'orientation': orientation_df,
        'tiles': tile_df,
        'num_frames': len(orientation_df)
    }


def load_content_features(video_name: str) -> dict:
    """
    Convenience function to load saliency and motion videos.

    Args:
        video_name: Name of the video

    Returns:
        Dictionary with saliency and motion videos
    """
    loader = DataLoader()

    saliency = loader.load_saliency_video(video_name)
    motion = loader.load_motion_video(video_name)

    return {
        'video_name': video_name,
        'saliency': saliency,
        'motion': motion,
        'num_frames': len(saliency)
    }


if __name__ == "__main__":
    # Test data loading
    print("=== Testing DataLoader ===\n")

    loader = DataLoader()

    # Test 1: Load orientation data
    print("Test 1: Loading orientation data...")
    video = "coaster2"
    user = 1
    try:
        orientation_df = loader.load_orientation_data(video, user)
        print(f"  Loaded {len(orientation_df)} frames for {video}, user {user}")
        print(f"  Columns: {list(orientation_df.columns)}")
        print(f"  Sample data:\n{orientation_df.head(3)}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Test 2: Load tile data
    print("Test 2: Loading tile data...")
    try:
        tile_df = loader.load_tile_data(video, user)
        print(f"  Loaded {len(tile_df)} frames")
        print(f"  Average tiles per frame: {tile_df['tile_numbers'].apply(len).mean():.1f}")
        print(f"  Sample data:\n{tile_df.head(3)}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Test 3: Load saliency video
    print("Test 3: Loading saliency video...")
    try:
        saliency = loader.load_saliency_video(video)
        print(f"  Loaded saliency video: shape {saliency.shape}")
        print(f"  Frame size: {saliency.shape[1]}x{saliency.shape[2]}")
        print(f"  Total frames: {saliency.shape[0]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Test 4: Load motion video
    print("Test 4: Loading motion video...")
    try:
        motion = loader.load_motion_video(video)
        print(f"  Loaded motion video: shape {motion.shape}")
        print(f"  Frame size: {motion.shape[1]}x{motion.shape[2]}")
        print(f"  Total frames: {motion.shape[0]}\n")
    except Exception as e:
        print(f"  Error: {e}\n")

    # Test 5: Get session info
    print("Test 5: Getting session info...")
    try:
        info = loader.get_session_info(video, user)
        print(f"  Session info:")
        for key, value in info.items():
            print(f"    {key}: {value}")
        print()
    except Exception as e:
        print(f"  Error: {e}\n")

    # Test 6: Get all sessions
    print("Test 6: Getting all sessions...")
    sessions = loader.get_all_sessions()
    print(f"  Total sessions: {len(sessions)}")
    print(f"  Sample sessions: {sessions[:5]}\n")

    print("=== DataLoader tests complete ===")
