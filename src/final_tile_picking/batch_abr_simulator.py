import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. CONFIGURATION
# ==========================================
# LOWER these bitrates to fit the network trace
BITRATE_LOW_KBPS = 200     # Background quality (was 500)
BITRATE_HIGH_KBPS = 1500   # Viewport quality (was 5000)

# KEEP these the same
TILE_ROWS = 4
TILE_COLS = 6
TOTAL_TILES = 24
HISTORY_WINDOW = 0.5
PREDICTION_WINDOW = 1.0

# Network Trace: Will be loaded from traces folder
# For now, we'll define a default but it will be overridden
NETWORK_TRACE = [3, 5, 8, 12, 15, 12, 8, 4]

# Path to traces folder (relative to this script location)
TRACES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traces')

# ==========================================
# 2. TRACE LOADING & MANAGEMENT
# ==========================================
def load_traces_from_folder(traces_dir):
    """Load all valid network traces from the traces folder."""
    traces = {}
    
    if not os.path.exists(traces_dir):
        print(f"WARNING: Traces directory not found: {traces_dir}")
        return traces
    
    # Load .mahi files (direct bandwidth values in Mbps)
    for filename in os.listdir(traces_dir):
        if filename.endswith('.mahi'):
            filepath = os.path.join(traces_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    values = [float(line.strip()) for line in f if line.strip()]
                
                # Skip if trace is too small (likely a test/demo trace)
                if len(values) < 20:
                    print(f"Skipped trace: {filename} ({len(values)} samples - too small)")
                    continue
                
                traces[filename] = values
                print(f"Loaded trace: {filename} ({len(values)} samples, avg={np.mean(values):.2f} Mbps)")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    
    # Load .dat files from subdirectories (cellular, fcc, hsdpa)
    # These are likely in Kbps, so convert to Mbps
    for subdir in ['cellular', 'fcc', 'hsdpa']:
        subdir_path = os.path.join(traces_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in sorted(os.listdir(subdir_path)):
                if filename.endswith('.dat'):
                    filepath = os.path.join(subdir_path, filename)
                    print(f"Loading {subdir}/{filename}...", end='', flush=True)
                    try:
                        # Load and parse more efficiently using numpy
                        with open(filepath, 'r') as f:
                            values = []
                            for line in f:
                                line = line.strip()
                                if line and line[0].isdigit():  # Only parse lines starting with a digit
                                    try:
                                        val = float(line)
                                        # Assume .dat files are in Kbps; convert to Mbps if needed
                                        if val > 1000:
                                            val = val / 1000.0
                                        values.append(val)
                                    except ValueError:
                                        pass  # Skip non-numeric lines
                            
                            if len(values) < 20:
                                print(f" SKIP ({len(values)} samples)")
                                continue
                            
                            trace_name = f"{subdir}/{filename}"
                            traces[trace_name] = values
                            avg_val = np.mean(values)
                            print(f" LOAD ({len(values)} samples, avg={avg_val:.2f} Mbps)")
                    except Exception as e:
                        print(f" ERROR: {e}")
    
    return traces

# ==========================================
# 3. CORE LOGIC
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

def run_simulation_on_file(user_trace_path, network_trace):
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
            bw_mbps = network_trace[int(t) % len(network_trace)]
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
# 4. DEBUG BATCH LOOP
# ==========================================
def run_batch_simulation(root_dir, network_traces):
    print(f"DEBUG: Starting search in: {root_dir}")
    print(f"DEBUG: Using {len(network_traces)} network traces")
    
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
                
                print(f"Processing: {filename}...")
                
                # Run simulation with each network trace
                for trace_name, trace_values in network_traces.items():
                    print(f"  - Using trace: {trace_name}...", end='\r')
                    sim_result = run_simulation_on_file(full_path, trace_values)
                    
                    if sim_result:
                        sim_result['user_file'] = filename
                        sim_result['network_trace'] = trace_name
                        results.append(sim_result)
    
    print(f"\nDEBUG: Found and processed {files_checked} valid files with {len(network_traces)} traces each.")

    if len(results) > 0:
        df = pd.DataFrame(results)
        print("-" * 80)
        print("FINAL RESULTS FOR REPORT:")
        print(f"Total simulations run: {len(results)}")
        print(f"Average Bandwidth Savings: {df['savings'].mean():.2%}")
        print(f"Average Stall Duration:    {df['stalls'].mean():.2f} seconds")
        print("-" * 80)
        print("\nResults by network trace:")
        for trace_name in df['network_trace'].unique():
            trace_df = df[df['network_trace'] == trace_name]
            print(f"  {trace_name}:")
            print(f"    Avg Savings: {trace_df['savings'].mean():.2%}")
            print(f"    Avg Stalls:  {trace_df['stalls'].mean():.2f}s")
        print("-" * 80)
        
        # Save to the CURRENT directory where you run the script from
        output_file = "final_abr_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved detailed results to: {os.path.abspath(output_file)}")
    else:
        print("No processed files found. Please run batch_process.py first.")

if __name__ == "__main__":
    # --- PATH CONFIGURATION ---
    DATA_DIR = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\data"
    
    # Load all available network traces
    print("Loading network traces from folder...")
    network_traces = load_traces_from_folder(TRACES_DIR)
    
    if not network_traces:
        print("ERROR: No network traces found. Using default trace.")
        network_traces = {'default': NETWORK_TRACE}
    
    print(f"Loaded {len(network_traces)} network traces.\n")
    
    run_batch_simulation(DATA_DIR, network_traces)