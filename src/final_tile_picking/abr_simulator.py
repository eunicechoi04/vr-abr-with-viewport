import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# CONFIGURATION
# ==========================================
# ASK PARTNER FOR THESE NUMBERS LATER
BITRATE_LOW_KBPS = 500    # Background quality
BITRATE_HIGH_KBPS = 5000  # Viewport quality

TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
HISTORY_WINDOW = 0.5
PREDICTION_WINDOW = 1.0

def get_tile_id(lat, lon):
    row = int(np.floor((np.clip(lat, -1.57, 1.57) + 1.57) / (3.14 / TILE_ROWS)))
    col = int(np.floor(((lon + 3.14) % 6.28) - 3.14 + 3.14) / (6.28 / TILE_COLS))
    return max(0, min(row, 3)) * TILE_COLS + max(0, min(col, 5))

def predict_future_tile(current_time, history_df):
    """Real Linear Regression Prediction"""
    # Get recent history
    mask = (history_df['timestamp'] >= current_time - HISTORY_WINDOW) & \
           (history_df['timestamp'] <= current_time)
    subset = history_df[mask]
    
    if len(subset) < 5:
        return None # Not enough data
        
    # Fit Line
    X = subset['timestamp'].values.reshape(-1, 1)
    model_lat = LinearRegression().fit(X, subset['latitude_rad'].values)
    model_lon = LinearRegression().fit(X, subset['longitude_rad'].values)
    
    # Predict Future
    future_time = np.array([[current_time + PREDICTION_WINDOW]])
    p_lat = model_lat.predict(future_time)[0]
    p_lon = model_lon.predict(future_time)[0]
    
    return get_tile_id(p_lat, p_lon)

def run_simulation(user_trace_path):
    print(f"Simulating ABR for: {user_trace_path}...")
    df = pd.read_csv(user_trace_path)
    
    # Network Trace (Simulating fluctuating 4G: 3Mbps to 15Mbps)
    network_trace = [3, 5, 8, 12, 15, 12, 8, 4] # Mbps
    
    buffer = 0.0
    stalls = 0.0
    total_bits = 0
    
    # Simulate every 1 second segment
    max_time = df['timestamp'].max()
    for t in np.arange(1.0, max_time - 1.0, 1.0):
        
        # 1. PREDICT
        pred_tile = predict_future_tile(t, df)
        
        # 2. DECIDE (Fetch Predict + Neighbors)
        # If prediction fails (None), fetch WHOLE 360 (Monolithic fallback)
        if pred_tile is None:
            download_size_bits = TOTAL_TILES * BITRATE_HIGH_KBPS * 1000
        else:
            # Fetch 1 High Quality Tile + 23 Low Quality Tiles
            # (In a real system, you'd fetch neighbors too, say 4 High, 20 Low)
            download_size_bits = (4 * BITRATE_HIGH_KBPS * 1000) + \
                                 (20 * BITRATE_LOW_KBPS * 1000)
        
        total_bits += download_size_bits
        
        # 3. DOWNLOAD
        bw_mbps = network_trace[int(t) % len(network_trace)]
        download_time = download_size_bits / (bw_mbps * 1_000_000)
        
        # 4. BUFFER LOGIC
        if download_time > 1.0:
            stalls += (download_time - 1.0)
        else:
            buffer += (1.0 - download_time)

    # METRICS
    baseline_bits = max_time * TOTAL_TILES * BITRATE_HIGH_KBPS * 1000
    savings = 1.0 - (total_bits / baseline_bits)
    
    return savings, stalls

if __name__ == "__main__":
    # Test on one file
    # REPLACE WITH ONE OF YOUR REAL FILES
    test_file = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\data\test0\Diving-2OzlksZBTiA\Diving-2OzlksZBTiA_0_processed.csv"
    
    sav, stall = run_simulation(test_file)
    print(f"\nRESULTS:")
    print(f"Bandwidth Savings: {sav:.2%}")
    print(f"Total Stall Time: {stall:.2f}s")