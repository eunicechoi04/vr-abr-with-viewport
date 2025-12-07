import pandas as pd
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def process_flare_data(input_path, output_path):
    # ---------------------------------------------------------
    # 1. LOAD RAW DATA
    # ---------------------------------------------------------
    # Your sample data format: Timestamp | FrameID | x | y | z | w
    # We infer this order because the last column is ~0.99 (the scalar w).
    print(f"Loading data from {input_path}...")
    
    try:
        # Load space-separated data, no header
        df = pd.read_csv(input_path, sep=r'\s+', header=None, 
                         names=['timestamp', 'frame_id', 'qx', 'qy', 'qz', 'qw'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Extract quaternions (x, y, z, w) and timestamps
    # SciPy natively expects (x, y, z, w), so no column reordering is needed for this dataset
    raw_quats = df[['qx', 'qy', 'qz', 'qw']].to_numpy()
    raw_times = df['timestamp'].to_numpy()

    # ---------------------------------------------------------
    # 2. RESAMPLING TO 100Hz (Section 3.1)
    # ---------------------------------------------------------
    # The paper states: "We convert the raw data... using a downsampling rate of 100Hz".
    TARGET_FREQ = 100  # Hz
    
    print(f"Resampling to {TARGET_FREQ}Hz using SLERP...")
    
    # Create the Rotation object
    rotations = R.from_quat(raw_quats)
    
    # Initialize SLERP (Spherical Linear Interpolation)
    # This handles the non-linear nature of 3D rotations correctly
    slerp = Slerp(raw_times, rotations)
    
    # Generate new monotonic timestamps at 100Hz (0.01s intervals)
    max_time = raw_times[-1]
    target_times = np.arange(0, max_time, 1/TARGET_FREQ)
    
    # Interpolate rotations at the new times
    interp_rots = slerp(target_times)

    # ---------------------------------------------------------
    # 3. CONVERT TO LATITUDE / LONGITUDE (Section 3.1)
    # ---------------------------------------------------------
    # The paper focuses on Pitch and Yaw.
    # In standard VR coordinate systems:
    # Yaw   = Longitude (Rotation around Y-axis)
    # Pitch = Latitude  (Rotation around X-axis)
    
    # We convert to Euler angles using 'yxz' sequence (Yaw, Pitch, Roll)
    euler_angles = interp_rots.as_euler('yxz', degrees=False)
    
    # Extract components
    yaw_long = euler_angles[:, 0]  # Longitude (-pi to pi)
    pitch_lat = euler_angles[:, 1] # Latitude  (-pi/2 to pi/2)
    
    # Note: The paper ignores roll, so we don't include it in analysis,
    # but we generate the lat/long columns they use for the User Study.

    # ---------------------------------------------------------
    # 4. FORMAT AND SAVE
    # ---------------------------------------------------------
    processed_df = pd.DataFrame({
        'timestamp': target_times,
        'latitude_rad': pitch_lat,
        'longitude_rad': yaw_long,
        # Adding degrees for easier validation
        'latitude_deg': np.degrees(pitch_lat),
        'longitude_deg': np.degrees(yaw_long)
    })
    
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed {len(processed_df)} samples.")
    print(f"Saved to: {output_path}")
    print("-" * 30)
    print(processed_df.head())

if __name__ == "__main__":
    # Point this to your specific text file
    INPUT_FILE = "C:\\Users\\feido\\Documents\\Code\\6.5820\\vr-abr-with-viewport\\360_Video_analysis\\data\\uid-3ba968b8-887c-460e-a5f2-86295957d731\\test0\\Diving-2OzlksZBTiA\\Diving-2OzlksZBTiA_0.txt" 
    OUTPUT_FILE = "flare_processed_100hz.csv"
    
    process_flare_data(INPUT_FILE, OUTPUT_FILE)