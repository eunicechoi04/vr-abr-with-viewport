import matplotlib.pyplot as plt
import numpy as np
from src.loader import load_head_movement_log
from src.analysis import resample_data, get_viewing_direction

# --- CONFIGURATION ---
# Point this to one of your downloaded txt files
FILE_PATH = "\\vr-abr-with-viewport\\360_Video_analysis\\data\\uid-3ba968b8-887c-460e-a5f2-86295957d731\\test0\\Diving-2OzlksZBTiA\\Diving-2OzlksZBTiA_0.txt" 

def main():
    print("1. Loading Data...")
    try:
        df = load_head_movement_log(FILE_PATH)
        print(f"   Loaded {len(df)} samples.")
    except FileNotFoundError:
        print("   Error: File not found. Please check the FILE_PATH in main.py")
        return

    print("2. Resampling to 30Hz using SLERP (Paper Section 5.1)...")
    resampled_df, rotations = resample_data(df)
    print(f"   Resampled to {len(resampled_df)} data points.")

    print("3. Calculating Viewing Directions (Longitude/Latitude)...")
    longitudes, latitudes = get_viewing_direction(rotations)

    print("4. Generating Heatmap (Replicating Figure 7)...")
    plt.figure(figsize=(10, 5))
    
    # Plot standard 2D histogram for equirectangular projection
    # Longitude: -pi to pi, Latitude: -pi/2 to pi/2
    plt.hist2d(longitudes, latitudes, bins=[60, 30], cmap='YlOrBr', range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]])
    
    plt.colorbar(label='Frequency')
    plt.title('User Viewing Probability (Replicating Fig 7)')
    plt.xlabel('Longitude (Radians)')
    plt.ylabel('Latitude (Radians)')
    plt.grid(True, alpha=0.3)
    
    # Save the output
    plt.savefig('heatmap_output.png')
    print("   Done! Check 'heatmap_output.png'.")
    plt.show()

if __name__ == "__main__":
    main()