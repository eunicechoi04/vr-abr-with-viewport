#!/usr/bin/env python3
"""Quick test of trace loading"""
import os
import sys
import numpy as np

# Add the final_tile_picking to path
sys.path.insert(0, 'src/final_tile_picking')

traces_dir = 'data/traces'

print(f"Checking directory: {os.path.abspath(traces_dir)}")
print(f"Directory exists: {os.path.exists(traces_dir)}")
print()

# Test .mahi files
print("=== .mahi files ===")
for filename in os.listdir(traces_dir):
    if filename.endswith('.mahi'):
        filepath = os.path.join(traces_dir, filename)
        with open(filepath, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        print(f"{filename}: {len(values)} samples, values={values}")
        if len(values) < 20:
            print(f"  -> SKIP: too small")
        else:
            print(f"  -> LOAD: avg={np.mean(values):.2f} Mbps")

# Test .dat files
print("\n=== .dat files ===")
for subdir in ['cellular', 'fcc', 'hsdpa']:
    subdir_path = os.path.join(traces_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"\n{subdir}:")
        for filename in os.listdir(subdir_path):
            if filename.endswith('.dat'):
                filepath = os.path.join(subdir_path, filename)
                values = []
                for line in open(filepath, 'r'):
                    line = line.strip()
                    if line and line[0].isdigit():
                        try:
                            val = float(line)
                            if val > 1000:
                                val = val / 1000.0
                            values.append(val)
                        except ValueError:
                            pass
                
                print(f"  {filename}: {len(values)} samples", end='')
                if len(values) > 0:
                    print(f", avg={np.mean(values):.2f} Mbps", end='')
                if len(values) < 20:
                    print(f" -> SKIP")
                else:
                    print(f" -> LOAD")
