import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# CONFIGURATION (Matches Flare Paper)
# ==========================================
HISTORY_WINDOW = 0.5      # Look at past 0.5s
PREDICTION_WINDOW = 1.0   # Predict 1.0s into the future
TILE_ROWS = 4
TILE_COLS = 6

def get_tile_id(lat_rad, lon_rad):
    """Maps coordinates to a specific Tile ID (0-23) for a 4x6 grid."""
    # Clip latitude to valid range [-pi/2, pi/2]
    lat_rad = np.clip(lat_rad, -np.pi/2, np.pi/2)
    # Wrap longitude to valid range [-pi, pi]
    lon_rad = (lon_rad + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate Row and Col
    row = int(np.floor((lat_rad + np.pi/2) / (np.pi / TILE_ROWS)))
    col = int(np.floor((lon_rad + np.pi) / (2*np.pi / TILE_COLS)))
    
    # Clamp to ensure we don't go out of bounds
    row = max(0, min(row, TILE_ROWS - 1))
    col = max(0, min(col, TILE_COLS - 1))
    
    return row * TILE_COLS + col

def evaluate_single_user(csv_path):
    """Runs Linear Regression on one user trace and returns accuracy."""
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure data is sorted by time
        df = df.sort_values('timestamp')
        
        timestamps = df['timestamp'].values
        lats = df['latitude_rad'].values
        lons = df['longitude_rad'].values
        
        # Define simulation boundaries
        start_time = timestamps[0] + HISTORY_WINDOW
        end_time = timestamps[-1] - PREDICTION_WINDOW
        
        hits = 0
        total = 0
        
        # Step through time (simulating a video player updating every 100ms)
        current_time = start_time
        while current_time < end_time:
            # 1. GET HISTORY (Past 0.5s)
            mask_hist = (timestamps >= (current_time - HISTORY_WINDOW)) & (timestamps <= current_time)
            
            # 2. GET GROUND TRUTH (Future 1.0s)
            target_time = current_time + PREDICTION_WINDOW
            # Find index of timestamp closest to target_time
            idx_future = (np.abs(timestamps - target_time)).argmin()
            
            # Skip if we don't have enough history data points to fit a line
            if np.sum(mask_hist) < 5:
                current_time += 0.1
                continue
                
            # 3. TRAIN MODEL (Online Linear Regression)
            X_hist = timestamps[mask_hist].reshape(-1, 1)
            y_lat = lats[mask_hist]
            y_lon = lons[mask_hist]
            
            # Fit separate models for Latitude (Pitch) and Longitude (Yaw)
            # Flare paper notes LR is sufficient for short windows
            model_lat = LinearRegression().fit(X_hist, y_lat)
            model_lon = LinearRegression().fit(X_hist, y_lon)
            
            # 4. PREDICT
            future_input = np.array([[target_time]])
            pred_lat = model_lat.predict(future_input)[0]
            pred_lon = model_lon.predict(future_input)[0]
            
            # 5. CHECK ACCURACY (Did we hit the right tile?)
            actual_lat = lats[idx_future]
            actual_lon = lons[idx_future]
            
            pred_tile = get_tile_id(pred_lat, pred_lon)
            actual_tile = get_tile_id(actual_lat, actual_lon)
            
            if pred_tile == actual_tile:
                hits += 1
            total += 1
            
            # Move forward 0.1s
            current_time += 0.1
            
        accuracy = hits / total if total > 0 else 0.0
        return accuracy
    
    except Exception as e:
        print(f"Error processing {os.path.basename(csv_path)}: {e}")
        return None

def run_benchmark(root_dir):
    print(f"Starting Prediction Benchmark on: {root_dir}")
    print("-" * 60)
    
    results = []
    
    # Walk through folders
    for current_root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Look only for the files you generated in the last step
            if filename.endswith("_processed.csv"):
                full_path = os.path.join(current_root, filename)
                
                print(f"Evaluating: {filename}...", end="\r")
                acc = evaluate_single_user(full_path)
                
                if acc is not None:
                    results.append({
                        'file': filename,
                        'accuracy': acc,
                        'parent_folder': os.path.basename(current_root) # usually video name
                    })
    
    print("\n" + "-" * 60)
    
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("final_prediction_results.csv", index=False)
    
    # Print Summary Stats
    print(f"Total Traces Evaluated: {len(results_df)}")
    print(f"Average Accuracy: {results_df['accuracy'].mean():.2%}")
    print(f"Min Accuracy: {results_df['accuracy'].min():.2%}")
    print(f"Max Accuracy: {results_df['accuracy'].max():.2%}")
    print("-" * 60)
    print("Saved detailed results to 'final_prediction_results.csv'")

if __name__ == "__main__":
    # Update this path to your data folder
    DATA_ROOT_DIR = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\data"
    run_benchmark(DATA_ROOT_DIR)