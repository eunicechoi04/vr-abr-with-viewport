"""
Quick test script to verify the tile ABR environment works correctly.

This tests the core simulation without running full experiments.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from final_tile_picking.tile_abr_env import (
            TileABREnv,
            run_single_experiment,
            TOTAL_TILES,
            SEGMENT_DURATION
        )
        print("✓ tile_abr_env imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_environment_creation():
    """Test creating an environment with sample data"""
    print("\nTesting environment creation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample user trace data
        timestamps = np.linspace(0, 10, 300)  # 10 seconds at 30 Hz
        latitudes = np.random.randn(300) * 0.1
        longitudes = np.random.randn(300) * 0.1
        tile_ids = np.random.randint(0, 24, 300)
        
        user_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude_rad': latitudes,
            'longitude_rad': longitudes,
            'tile_id': tile_ids
        })
        
        # Create sample network trace
        network_trace = [5.0] * 10  # 5 Mbps for 10 seconds
        
        # Create environment
        from final_tile_picking.tile_abr_env import TileABREnv
        
        env = TileABREnv(
            user_trace_df=user_df,
            network_trace_mbps=network_trace,
            motion_scores=None,
            saliency_scores=None,
            num_high_quality_tiles=4
        )
        
        print(f"✓ Environment created successfully")
        print(f"  - Video duration: {env.video_duration:.1f}s")
        print(f"  - Number of segments: {env.num_segments}")
        print(f"  - Buffer: {env.buffer}s")
        
        return env
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction():
    """Test tile prediction from user data"""
    print("\nTesting tile prediction...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data with clear movement pattern
        timestamps = np.linspace(0, 2, 60)  # 2 seconds
        latitudes = timestamps * 0.1  # Linear movement
        longitudes = timestamps * 0.05
        
        user_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude_rad': latitudes,
            'longitude_rad': longitudes,
            'tile_id': np.zeros(60, dtype=int)
        })
        
        network_trace = [5.0] * 2
        
        from final_tile_picking.tile_abr_env import TileABREnv
        
        env = TileABREnv(
            user_trace_df=user_df,
            network_trace_mbps=network_trace
        )
        
        # Test prediction at t=1.0
        predicted_tile = env.predict_next_tile(1.0)
        
        if predicted_tile is not None:
            print(f"✓ Prediction successful: tile {predicted_tile}")
        else:
            print(f"✓ Prediction returned None (not enough history)")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulation_step():
    """Test running one simulation step"""
    print("\nTesting simulation step...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        timestamps = np.linspace(0, 5, 150)  # 5 seconds
        user_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude_rad': np.random.randn(150) * 0.1,
            'longitude_rad': np.random.randn(150) * 0.1,
            'tile_id': np.random.randint(0, 24, 150)
        })
        
        network_trace = [5.0] * 5
        
        from final_tile_picking.tile_abr_env import TileABREnv
        
        env = TileABREnv(
            user_trace_df=user_df,
            network_trace_mbps=network_trace
        )
        
        # Run one step
        high_quality_tiles = [10, 11, 13, 14]  # Select 4 tiles
        result = env.step(high_quality_tiles)
        
        print(f"✓ Simulation step successful")
        print(f"  - Bandwidth: {result['bandwidth_mbps']} Mbps")
        print(f"  - Download time: {result['download_time']:.3f}s")
        print(f"  - Rebuffer time: {result['rebuf_time']:.3f}s")
        print(f"  - Buffer level: {result['buffer_level']:.3f}s")
        print(f"  - Tiles hit: {len(result['tiles_hit'])}")
        print(f"  - Tiles missed: {len(result['tiles_missed'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simulation step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_simulation():
    """Test running a complete simulation"""
    print("\nTesting full simulation...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create 10-second trace
        timestamps = np.linspace(0, 10, 300)
        user_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude_rad': np.sin(timestamps * 0.5) * 0.3,  # Oscillating movement
            'longitude_rad': np.cos(timestamps * 0.5) * 0.3,
            'tile_id': np.random.randint(0, 24, 300)
        })
        
        network_trace = [5.0, 8.0, 3.0, 10.0, 6.0, 4.0, 7.0, 9.0, 5.0, 6.0]
        
        from final_tile_picking.tile_abr_env import run_single_experiment
        
        # Create a temp CSV file
        temp_csv = Path("/tmp/test_user_trace.csv")
        user_df.to_csv(temp_csv, index=False)
        
        # Run simulation
        result = run_single_experiment(
            user_trace_path=str(temp_csv),
            network_trace_mbps=network_trace,
            approach='predictive_with_content'
        )
        
        print(f"✓ Full simulation successful")
        print(f"  - Approach: {result['approach']}")
        print(f"  - Segments: {result['num_segments']}")
        print(f"  - Total QoE: {result['total_qoe']:.2f}")
        print(f"  - Avg QoE per segment: {result['avg_qoe_per_segment']:.2f}")
        print(f"  - Total rebuffering: {result['total_rebuf_time']:.2f}s")
        print(f"  - Viewport hit rate: {result['viewport_hit_rate']:.2%}")
        print(f"  - Bandwidth savings: {result['bandwidth_savings']:.2%}")
        print(f"  - Total MB downloaded: {result['total_mb_downloaded']:.2f} MB")
        
        # Clean up
        temp_csv.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Full simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("TILE ABR ENVIRONMENT TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Import test", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Environment creation", test_environment_creation() is not None))
        results.append(("Tile prediction", test_prediction()))
        results.append(("Simulation step", test_simulation_step()))
        results.append(("Full simulation", test_full_simulation()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The tile ABR environment is working correctly.")
        print("\nNext steps:")
        print("1. Install pandas if not available: pip install pandas numpy scikit-learn")
        print("2. Run full experiments: python3 src/run_tile_experiments.py")
        print("3. Check NETWORK_SIMULATION_GUIDE.md for details")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")


if __name__ == "__main__":
    main()
