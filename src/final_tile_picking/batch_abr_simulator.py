import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURATION
# ==========================================
BITRATE_LOW_KBPS = 500
BITRATE_HIGH_KBPS = 5000
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
HISTORY_WINDOW = 0.5
PREDICTION_WINDOW = 1.0
NETWORK_TRACE = [3, 5, 8, 12, 15, 12, 8, 4] 

# ==========================================
# 2. CORE LOGIC
# ==========================================
def get_tile_id(lat, lon):
    row = int(np.floor((np.clip(lat, -1.57, 1.57) + 1.57) / (3.14 / TILE_ROWS)))
    col = int(np.floor(((lon + 3.14) % 6.28) - 3.14 + 3.14) / (6.28 / TILE_COLS))
    return max(0, min(row, 3)) * TILE_COLS + max(0, min(col, 5))

def predict_future_tile(current_time, history_df):
    mask = (history_df['timestamp'] >= current_time - HISTORY_WINDOW) & \
           (history_df['timestamp'] <= current_time)
    subset = history_df[mask]
    if len(subset) < 5: return None 
        
    X = subset['timestamp'].values.reshape(-1, 1)
    model_lat = LinearRegression().fit(X, subset['latitude_rad'].values)
    model_lon = LinearRegression().fit(X, subset['longitude_rad'].values)
    
    future_time = np.array([[current_time + PREDICTION_WINDOW]])
    p_lat = model_lat.predict(future_time)[0]
    p_lon = model_lon.predict(future_time)[0]
    return get_tile_id(p_lat, p_lon)

def run_simulation_on_file(user_trace_path):
    try:
        df = pd.read_csv(user_trace_path)
        buffer = 0.0
        stalls = 0.0
        total_bits_downloaded = 0
        max_time = df['timestamp'].max()
        segments = np.arange(1.0, max_time - 1.0, 1.0)
        
        if len(segments) == 0: return None
            
        for t in segments:
            pred_tile = predict_future_tile(t, df)
            
            if pred_tile is None:
                download_size_bits = TOTAL_TILES * BITRATE_HIGH_KBPS * 1000
            else:
                download_size_bits = (4 * BITRATE_HIGH_KBPS * 1000) + \
                                     (20 * BITRATE_LOW_KBPS * 1000)
            
            total_bits_downloaded += download_size_bits
            bw_mbps = NETWORK_TRACE[int(t) % len(NETWORK_TRACE)]
            download_time = download_size_bits / (bw_mbps * 1_000_000)
            
            if download_time > 1.0:
                stalls += (download_time - 1.0)
            else:
                buffer += (1.0 - download_time)

        baseline_bits = len(segments) * TOTAL_TILES * BITRATE_HIGH_KBPS * 1000
        bandwidth_savings = 1.0 - (total_bits_downloaded / baseline_bits)
        return {'savings': bandwidth_savings, 'stalls': stalls}

    except Exception as e:
        return None

# ==========================================
# 3. DEBUG BATCH LOOP
# ==========================================
def run_batch_simulation(root_dir):
    print(f"DEBUG: Starting search in: {root_dir}")
    
    if not os.path.exists(root_dir):
        print("ERROR: The path provided does not exist!")
        return

    results = []
    files_checked = 0
    
    # Crawler
    for current_root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Check strictly for _processed.csv
            if filename.endswith("_processed.csv"):
                full_path = os.path.join(current_root, filename)
                files_checked += 1
                
                print(f"Running: {filename}...", end='\r')
                sim_result = run_simulation_on_file(full_path)
                
                if sim_result:
                    results.append(sim_result)
    
    print(f"\nDEBUG: Found and processed {files_checked} valid files.")

    if len(results) > 0:
        df = pd.DataFrame(results)
        print("-" * 60)
        print("FINAL RESULTS FOR REPORT:")
        print(f"Average Bandwidth Savings: {df['savings'].mean():.2%}")
        print(f"Average Stall Duration:    {df['stalls'].mean():.2f} seconds")
        print("-" * 60)
        
        # Save to the CURRENT directory where you run the script from
        output_file = "final_abr_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved detailed results to: {os.path.abspath(output_file)}")
    else:
        print("No processed files found. Please run batch_process.py first.")

if __name__ == "__main__":
    # --- PATH CONFIGURATION ---
    # I'm using the exact path from your error message
    DATA_DIR = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\data"
    
    run_batch_simulation(DATA_DIR)