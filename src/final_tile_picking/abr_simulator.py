import pandas as pd
import numpy as np
import json

# --- CONFIGURATION ---
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
SEGMENT_DURATION = 1.0  # seconds

# Bitrates in Kilobits per second (Kbps)
BITRATE_LOW = 500
BITRATE_HIGH = 5000 

def get_predicted_tiles(current_time, history_df):
    """
    Your Linear Regression Logic goes here.
    For now, we can use a placeholder: 'Last Known Position' (Static Prediction)
    if you haven't wrapped your LR logic into a function yet.
    """
    # 1. Get data up to current_time
    past_data = history_df[history_df['timestamp'] <= current_time]
    if past_data.empty:
        return [0] # Default to tile 0 if no data
    
    # 2. Get last known tile (Static Prediction is a good baseline)
    last_row = past_data.iloc[-1]
    center_tile = int(last_row['tile_id'])
    
    # 3. Add neighbors (Margin of Error)
    # A simple 3x3 grid around the center tile
    predicted_tiles = {center_tile}
    
    # Logic to add adjacent tiles would go here
    # For simulation simplicity, let's assume we fetch 4 tiles (Viewport area)
    return list(predicted_tiles)

def run_simulation(user_trace_file, network_trace_mbps):
    print(f"Simulating: {user_trace_file}")
    df = pd.read_csv(user_trace_file)
    
    # Simulation State
    buffer_level = 0.0
    total_bits_downloaded = 0
    total_stall_time = 0.0
    video_duration = df['timestamp'].max()
    
    # Iterate through video segments (0s, 1s, 2s...)
    for t in np.arange(0, video_duration, SEGMENT_DURATION):
        
        # 1. VIEWPORT PREDICTION
        predicted_tile_ids = get_predicted_tiles(t, df)
        
        # 2. ABR DECISION (Flare Logic)
        # High quality for predicted, Low for background
        num_high = len(predicted_tile_ids)
        num_low = TOTAL_TILES - num_high
        
        segment_size_bits = (num_high * (BITRATE_HIGH/TOTAL_TILES)) + \
                            (num_low * (BITRATE_LOW/TOTAL_TILES))
        
        # 3. NETWORK SIMULATION
        # Get bandwidth at this second (wrap around trace if needed)
        bandwidth_mbps = network_trace_mbps[int(t) % len(network_trace_mbps)]
        bandwidth_bps = bandwidth_mbps * 1_000_000
        
        download_time = segment_size_bits / bandwidth_bps
        total_bits_downloaded += segment_size_bits
        
        # 4. BUFFER & STALL CALCULATION
        if download_time > SEGMENT_DURATION:
            # We took too long! Buffer drains.
            deficit = download_time - SEGMENT_DURATION
            if buffer_level >= deficit:
                buffer_level -= deficit
            else:
                total_stall_time += (deficit - buffer_level)
                buffer_level = 0
        else:
            # We were fast! Buffer grows.
            buffer_level += (SEGMENT_DURATION - download_time)
            
    # RESULTS
    print(f"  Total Download: {total_bits_downloaded / 1_000_000:.2f} Mb")
    print(f"  Stall Time: {total_stall_time:.2f} sec")
    
    # Compare to Baseline (Monolithic - All High Quality)
    baseline_bits = (BITRATE_HIGH * video_duration)
    savings = 1.0 - (total_bits_downloaded / baseline_bits)
    print(f"  Bandwidth Savings: {savings:.2%}")

if __name__ == "__main__":
    # Example: Variable Network (5Mbps to 20Mbps)
    NETWORK_TRACE = [5, 10, 15, 20, 15, 10, 5, 2, 5, 10] 
    
    # Pick one processed file to test
    TEST_FILE = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\data\test_user_processed.csv"
    
    # Note: You need to point this to a real file path!
    # run_simulation(TEST_FILE, NETWORK_TRACE)