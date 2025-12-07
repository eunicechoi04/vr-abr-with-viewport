import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. CORE PROCESSING LOGIC
# ==========================================
# ==========================================
# 1. CORE PROCESSING LOGIC (Updated with Velocity and Tile ID)
# ==========================================
def process_single_file(input_path, output_path):
    try:
        # Load space-separated data
        df = pd.read_csv(input_path, sep=r'\s+', header=None, 
                         names=['timestamp', 'frame_id', 'qx', 'qy', 'qz', 'qw'])
        
        # Check if valid numeric data
        if not np.issubdtype(df['timestamp'].dtype, np.number):
            print(f"   [SKIP] {os.path.basename(input_path)} contains non-numeric data.")
            return

        # Extract vectors
        raw_quats = df[['qx', 'qy', 'qz', 'qw']].to_numpy()
        raw_times = df['timestamp'].to_numpy()

        # --- 100Hz Resampling (Flare Paper Method) ---
        TARGET_FREQ = 100
        rotations = R.from_quat(raw_quats)
        slerp = Slerp(raw_times, rotations)
        
        max_time = raw_times[-1]
        target_times = np.arange(0, max_time, 1/TARGET_FREQ)
        interp_rots = slerp(target_times)

        # --- Lat/Lon Conversion ---
        euler_angles = interp_rots.as_euler('yxz', degrees=False)
        yaw_long = euler_angles[:, 0]
        pitch_lat = euler_angles[:, 1]
        
        # ------------------------------------------------------------------
        # NEW FEATURE EXTRACTION CODE GOES HERE
        # ------------------------------------------------------------------
        
        # 1. Calculate Angular Velocity (radians/s)
        
        # Create a temporary DataFrame for easier calculation of differences
        temp_df = pd.DataFrame({
            'timestamp': target_times,
            'latitude_rad': pitch_lat,
            'longitude_rad': yaw_long
        })
        
        # Calculate change in time (dt) and change in position (d_lat, d_lon)
        # fillna(0) for the first row where diff is NaN
        temp_df['dt'] = temp_df['timestamp'].diff().fillna(0.01) 
        temp_df['delta_lat'] = temp_df['latitude_rad'].diff().fillna(0)
        temp_df['delta_lon'] = temp_df['longitude_rad'].diff().fillna(0)
        
        # Angular Velocity (change in angle / change in time)
        temp_df['velocity_lat'] = temp_df['delta_lat'] / temp_df['dt']
        temp_df['velocity_lon'] = temp_df['delta_lon'] / temp_df['dt']
        
        # 2. Map to Tiles (Assuming a 4x6 grid, common in literature)
        
        rows = 4
        cols = 6
        
        # Normalize latitude (-pi/2 to pi/2) to a 0-to-rows range
        temp_df['tile_row'] = np.floor((temp_df['latitude_rad'] + np.pi/2) / (np.pi / rows)).astype(int).clip(0, rows-1)
        # Normalize longitude (-pi to pi) to a 0-to-cols range
        temp_df['tile_col'] = np.floor((temp_df['longitude_rad'] + np.pi) / (2*np.pi / cols)).astype(int).clip(0, cols-1)
        
        # Create a single Tile ID
        temp_df['tile_id'] = temp_df['tile_row'] * cols + temp_df['tile_col']

        # ------------------------------------------------------------------
        
        # Create final output DataFrame using all columns from temp_df
        processed_df = pd.DataFrame({
            'timestamp': temp_df['timestamp'],
            'latitude_rad': temp_df['latitude_rad'],
            'longitude_rad': temp_df['longitude_rad'],
            'latitude_deg': np.degrees(temp_df['latitude_rad']),
            'longitude_deg': np.degrees(temp_df['longitude_rad']),
            'velocity_lat': temp_df['velocity_lat'],
            'velocity_lon': temp_df['velocity_lon'],
            'tile_row': temp_df['tile_row'],
            'tile_col': temp_df['tile_col'],
            'tile_id': temp_df['tile_id']
        })

        processed_df.to_csv(output_path, index=False)
        print(f"   [SUCCESS] Created: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"   [ERROR] Failed on {os.path.basename(input_path)}: {e}")

# ==========================================
# 2. DEBUG BATCH WALKING LOGIC
# ==========================================
def run_batch_processing(root_dir):
    print(f"STARTING SEARCH IN: {root_dir}")
    
    if not os.path.exists(root_dir):
        print("!!! ERROR: The directory path does not exist. Please check the path.")
        return

    files_found = 0
    files_processed = 0
    
    # Walk through every folder
    for current_root, dirs, files in os.walk(root_dir):
        # print(f"Scanning folder: {current_root}") # Uncomment if you want to see every folder
        
        for filename in files:
            files_found += 1
            
            # Filter Logic
            is_txt = filename.endswith(".txt")
            is_not_meta = "testInfo" not in filename and "formAnswers" not in filename
            
            if is_txt and is_not_meta:
                input_full_path = os.path.join(current_root, filename)
                output_filename = filename.replace(".txt", "_processed.csv")
                output_full_path = os.path.join(current_root, output_filename)
                
                print(f"-> Found Candidate: {filename}")
                process_single_file(input_full_path, output_full_path)
                files_processed += 1
            else:
                # Optional: See what is being ignored
                # print(f"   [Ignored] {filename}")
                pass

    print("-" * 50)
    print(f"SCAN COMPLETE.")
    print(f"Total files seen: {files_found}")
    print(f"Total files processed: {files_processed}")
    
    if files_processed == 0:
        print("\nTROUBLESHOOTING:")
        print("1. Did the path point to the 'data' folder?")
        print("2. Are the files actually .txt files?")

if __name__ == "__main__":
    # --- UPDATE THIS PATH ---
    # Based on your previous error, this is the likely path.
    # MAKE SURE there are no extra spaces.
    DATA_ROOT_DIR = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\data"
    
    run_batch_processing(DATA_ROOT_DIR)