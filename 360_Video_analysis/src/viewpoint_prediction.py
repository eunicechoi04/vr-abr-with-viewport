import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION ---
HISTORY_WINDOW = 0.5  # Seconds of history to look at
PREDICTION_WINDOW = 1.0  # Seconds into the future to predict
TILE_ROWS = 4
TILE_COLS = 6

def get_tile_id(lat_rad, lon_rad):
    """Maps a latitude/longitude to a Tile ID (0 to 23)."""
    # Clip latitude to [-pi/2, pi/2]
    lat_rad = np.clip(lat_rad, -np.pi/2, np.pi/2)
    # Wrap longitude to [-pi, pi]
    lon_rad = (lon_rad + np.pi) % (2 * np.pi) - np.pi
    
    row = int(np.floor((lat_rad + np.pi/2) / (np.pi / TILE_ROWS)))
    col = int(np.floor((lon_rad + np.pi) / (2*np.pi / TILE_COLS)))
    
    # Clamp to valid range
    row = max(0, min(row, TILE_ROWS - 1))
    col = max(0, min(col, TILE_COLS - 1))
    
    return row * TILE_COLS + col

def run_prediction(csv_path):
    print(f"Evaluating Prediction on: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # We need time to be monotonic for regression
    timestamps = df['timestamp'].values
    lats = df['latitude_rad'].values
    lons = df['longitude_rad'].values
    
    # Store results
    results = []
    
    # Start simulating after we have enough history
    # And stop before we run out of future data
    start_time = timestamps[0] + HISTORY_WINDOW
    end_time = timestamps[-1] - PREDICTION_WINDOW
    
    # Step through the video at 100ms intervals (typical ABR update rate)
    current_time = start_time
    
    hits = 0
    total_checks = 0
    
    while current_time < end_time:
        # 1. IDENTIFY WINDOWS
        # Past: [current_time - HISTORY_WINDOW, current_time]
        past_mask = (timestamps >= (current_time - HISTORY_WINDOW)) & (timestamps <= current_time)
        
        # Future: The exact moment at [current_time + PREDICTION_WINDOW]
        # We find the closest timestamp in the data to our target
        future_target_time = current_time + PREDICTION_WINDOW
        future_idx = (np.abs(timestamps - future_target_time)).argmin()
        
        if np.sum(past_mask) < 5: # Need at least a few points to fit a line
            current_time += 0.1
            continue
            
        # 2. TRAIN LINEAR REGRESSION (The "Brain")
        # X = Time, Y = Latitude/Longitude
        X_hist = timestamps[past_mask].reshape(-1, 1)
        y_lat_hist = lats[past_mask]
        y_lon_hist = lons[past_mask]
        
        # Fit models
        lat_model = LinearRegression().fit(X_hist, y_lat_hist)
        lon_model = LinearRegression().fit(X_hist, y_lon_hist)
        
        # 3. PREDICT FUTURE
        future_time_reshaped = np.array([[future_target_time]])
        pred_lat = lat_model.predict(future_time_reshaped)[0]
        pred_lon = lon_model.predict(future_time_reshaped)[0]
        
        # 4. EVALUATE (Hit or Miss?)
        actual_lat = lats[future_idx]
        actual_lon = lons[future_idx]
        
        pred_tile = get_tile_id(pred_lat, pred_lon)
        actual_tile = get_tile_id(actual_lat, actual_lon)
        
        is_hit = (pred_tile == actual_tile)
        
        results.append({
            'time': current_time,
            'pred_lat': pred_lat,
            'actual_lat': actual_lat,
            'is_hit': is_hit
        })
        
        if is_hit:
            hits += 1
        total_checks += 1
        
        current_time += 0.1 # Move forward 100ms
        
    accuracy = hits / total_checks if total_checks > 0 else 0
    print(f"Accuracy with {HISTORY_WINDOW}s history -> {PREDICTION_WINDOW}s future: {accuracy:.2%}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Point this to one of your PROCESSED files
    FILE = "path/to/your/Diving_processed.csv" 
    results_df = run_prediction(FILE)
    
    # Optional: Plot the "Trajectory"
    # plt.plot(results_df['time'], results_df['actual_lat'], label='Actual')
    # plt.plot(results_df['time'], results_df['pred_lat'], label='Predicted', linestyle='--')
    # plt.legend()
    # plt.show()