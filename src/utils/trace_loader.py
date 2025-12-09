"""
Network trace loader for various trace formats.
Supports .mahi, .dat, and .log trace files.
"""
import os
from pathlib import Path


def load_mahi_trace(file_path):
    """
    Load .mahi trace file.
    Format: One bandwidth value (Mbps) per line.

    Returns: List of bandwidth values in Mbps
    """
    with open(file_path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return values


def load_dat_trace(file_path, unit='kbps'):
    """
    Load .dat trace file (cellular traces).
    Format: One bandwidth value per line.

    Args:
        file_path: Path to .dat file
        unit: 'kbps' or 'mbps' - unit of values in file

    Returns: List of bandwidth values in Mbps
    """
    with open(file_path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]

    # Convert to Mbps if needed
    if unit.lower() == 'kbps':
        values = [v / 1000.0 for v in values]

    return values


def load_log_trace(file_path):
    """
    Load .log trace file (FCC traces).
    Format: One bandwidth value (Mbps) per line.

    Returns: List of bandwidth values in Mbps
    """
    with open(file_path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return values


def load_trace(file_path, unit='auto'):
    """
    Auto-detect format and load trace file.

    Args:
        file_path: Path to trace file
        unit: 'auto', 'kbps', or 'mbps'

    Returns: List of bandwidth values in Mbps
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == '.mahi':
        return load_mahi_trace(file_path)
    elif ext == '.dat':
        # .dat files from cellular traces are in Kbps
        return load_dat_trace(file_path, unit='kbps' if unit == 'auto' else unit)
    elif ext == '.log':
        return load_log_trace(file_path)
    else:
        # Try as generic text file with one value per line
        try:
            with open(file_path, 'r') as f:
                values = [float(line.strip()) for line in f if line.strip()]
            return values
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None


def load_all_traces_from_directory(trace_dir, max_traces=None, sample_every_n=1):
    """
    Load all trace files from a directory and subdirectories.

    Args:
        trace_dir: Root directory containing traces
        max_traces: Maximum number of traces to load (None = all)
        sample_every_n: Sample every Nth value to reduce trace length (1 = no sampling)

    Returns:
        Dictionary mapping trace names to bandwidth lists
    """
    trace_dir = Path(trace_dir)
    traces = {}

    # Find all trace files
    trace_files = []
    for ext in ['.mahi', '.dat', '.log']:
        trace_files.extend(trace_dir.rglob(f'*{ext}'))

    if max_traces:
        trace_files = trace_files[:max_traces]

    print(f"Found {len(trace_files)} trace files")

    for i, trace_path in enumerate(trace_files):
        # Create trace name from path
        trace_name = str(trace_path.relative_to(trace_dir)).replace('/', '_').replace('\\', '_')
        trace_name = trace_name.replace(trace_path.suffix, '')

        print(f"  [{i+1}/{len(trace_files)}] Loading {trace_name}...", end=' ')

        try:
            bandwidth_values = load_trace(trace_path)

            if bandwidth_values is None:
                print("FAILED")
                continue

            # Sample if requested
            if sample_every_n > 1:
                bandwidth_values = bandwidth_values[::sample_every_n]

            # Remove zeros and very small values (likely errors)
            bandwidth_values = [max(v, 0.1) for v in bandwidth_values]

            traces[trace_name] = bandwidth_values
            print(f"OK ({len(bandwidth_values)} samples)")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nSuccessfully loaded {len(traces)} traces")
    return traces


def get_summary_statistics(traces):
    """
    Get summary statistics for loaded traces.

    Args:
        traces: Dictionary of trace name -> bandwidth list

    Returns:
        Dictionary with statistics
    """
    import numpy as np

    stats = {}

    for trace_name, bandwidth_values in traces.items():
        bw_array = np.array(bandwidth_values)

        stats[trace_name] = {
            'min_mbps': float(bw_array.min()),
            'max_mbps': float(bw_array.max()),
            'mean_mbps': float(bw_array.mean()),
            'median_mbps': float(np.median(bw_array)),
            'std_mbps': float(bw_array.std()),
            'samples': len(bandwidth_values)
        }

    return stats

def expand_values(values, batch_size):
    n = len(values)
    # repeats per value
    r = batch_size // n
    
    expanded = []
    for v in values:
        expanded.extend([v] * r)
    
    # If batch_size not divisible cleanly, fill remaining with last value
        # (optional behavior)
    while len(expanded) < batch_size:
        expanded.append(values[-1])
    
    return expanded