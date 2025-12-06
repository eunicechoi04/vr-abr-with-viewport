Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 360° VR ABR Streaming with Viewport Prediction - Implementation Plan

 Project Overview

 Goal: Research prototype demonstrating bandwidth savings (>50%) using viewport prediction + tiled ABR streaming for 360° VR
 videos, while maintaining high QoE.

 Approach: Simulation-based evaluation using existing dataset (10 videos, 500 user sessions) with a content-aware LSTM viewport
 predictor.

 Timeline: 7-14 days for working prototype with evaluation results.

 ---
 Project Structure

 /Users/eunicechoi04/Downloads/abr/
 ├── src/
 │   ├── config.py                         # Configuration constants
 │   ├── preprocessing/
 │   │   ├── data_loader.py                # Load CSVs and videos ⭐ CRITICAL
 │   │   ├── tile_encoder.py               # FFmpeg tile encoding
 │   │   ├── feature_extractor.py          # Extract saliency/motion features
 │   │   └── dataset_builder.py            # Build train/test datasets
 │   ├── models/
 │   │   ├── baseline.py                   # Baseline predictors
 │   │   ├── lstm_predictor.py             # LSTM viewport prediction ⭐ CRITICAL
 │   │   ├── attention_predictor.py        # Content-aware LSTM+Attention
 │   │   └── trainer.py                    # Training utilities
 │   ├── simulation/
 │   │   ├── bandwidth_model.py            # Network simulation
 │   │   ├── tile_selector.py              # ABR tile selection ⭐ CRITICAL
 │   │   ├── buffer_manager.py             # Video buffer simulation
 │   │   └── session_simulator.py          # Replay user sessions
 │   ├── evaluation/
 │   │   ├── metrics.py                    # All metrics ⭐ CRITICAL
 │   │   ├── visualizer.py                 # Result plots
 │   │   └── evaluator.py                  # Evaluation pipeline
 │   └── utils/
 │       ├── viewport.py                   # Viewport geometry ⭐ CRITICAL
 │       └── logger.py                     # Logging
 ├── scripts/
 │   ├── 01_preprocess_data.py             # Data preprocessing
 │   ├── 02_encode_tiles.py                # Tile encoding (optional)
 │   ├── 03_train_predictor.py             # Train model
 │   ├── 04_run_simulation.py              # ABR simulation
 │   └── 05_evaluate_results.py            # Evaluation
 ├── notebooks/
 │   ├── 01_data_exploration.ipynb         # Dataset exploration
 │   └── 02_results_analysis.ipynb         # Results analysis
 ├── outputs/
 │   ├── tiles/                            # Encoded videos (gitignored)
 │   ├── preprocessed/                     # Training data
 │   ├── models/                           # Model checkpoints
 │   ├── simulations/                      # Simulation logs
 │   └── plots/                            # Figures
 ├── tests/
 ├── requirements.txt
 ├── README.md
 └── .gitignore

 ---
 Implementation Phases

 Phase 1: Setup and Data Preprocessing (Days 1-2)

 Objective: Load and explore the dataset, build training data.

 Tasks:

 1. Environment Setup
   - Create requirements.txt:
   numpy>=1.21.0
 pandas>=1.3.0
 opencv-python>=4.5.0
 torch>=2.0.0
 torchvision>=0.15.0
 ffmpeg-python>=0.2.0
 scikit-image>=0.19.0
 scikit-video>=1.1.11
 matplotlib>=3.5.0
 seaborn>=0.11.0
 jupyter>=1.0.0
 h5py>=3.7.0
   - Create project structure
 2. Data Loader (src/preprocessing/data_loader.py) ⭐
 class DataLoader:
     def load_orientation_data(video_name, user_id) -> pd.DataFrame
         # Returns: frame_no, yaw, pitch, roll

     def load_tile_data(video_name, user_id) -> pd.DataFrame
         # Returns: frame_no, tile_numbers (list)

     def load_saliency_video(video_name) -> np.ndarray
         # Returns: [T, H, W, 3] video frames

     def load_motion_video(video_name) -> np.ndarray
         # Returns: [T, H, W, 3] optical flow frames

     def get_all_sessions() -> List[Tuple[str, int]]
         # Returns: [(video_name, user_id), ...]
   - Load from /Users/eunicechoi04/Downloads/abr/data/360dataset/
   - Parse CSV files with proper column names
 3. Viewport Utilities (src/utils/viewport.py) ⭐
 def orientation_to_tiles(yaw, pitch, roll, fov=110) -> List[int]
     # Convert orientation to visible tile IDs

 def tile_overlap_iou(predicted_tiles, actual_tiles) -> float
     # Calculate IoU between tile sets

 def tiles_to_bandwidth(tiles, quality_levels) -> int
     # Estimate bandwidth from tile selection
   - Assume 10×20 tile grid (200 tiles total for equirectangular)
   - Handle wrap-around at 360° and poles
 4. Dataset Builder (src/preprocessing/dataset_builder.py)
   - Convert CSVs to training sequences:
       - Input: Last N=30 frames (1s) of orientation + velocity
     - Output: Future orientation at t+30, t+60, t+90 (1s, 2s, 3s ahead)
   - Split strategy:
       - Leave-one-video-out: 9 videos train, 1 test
     - Per-video: 40 users train, 10 users test
   - Save as HDF5 files in /outputs/preprocessed/
 5. Exploration Notebook (notebooks/01_data_exploration.ipynb)
   - Visualize viewport trajectories
   - Analyze tile distributions
   - Check saliency/motion features

 Deliverables:
 - Working data loading pipeline
 - Preprocessed HDF5 files
 - Exploration notebook with insights

 ---
 Phase 2: Tile Encoding (Days 2-3) - OPTIONAL

 Objective: Encode videos into tiles at multiple quality levels.

 Quick Path (Recommended): Skip actual encoding, use bandwidth estimation:
 - High quality: 50 KB/tile
 - Medium quality: 25 KB/tile
 - Low quality: 10 KB/tile

 Full Path (If time permits):
 1. Tile Encoder (src/preprocessing/tile_encoder.py)
   - Use FFmpeg to crop equirectangular video into 10×20 tiles
   - Encode each tile at 3 quality levels (H.264 CRF: 18/23/28)
   - Create 2-second segments for ABR
   - Start with 2-3 videos (coaster, diving, game)
 2. Parallelization:
   - Use multiprocessing to encode tiles in parallel
   - Expected time: ~2-4 hours for 3 videos on modern CPU

 Deliverables:
 - Tile manifest JSON with tile metadata
 - Encoded tiles in /outputs/tiles/ OR bandwidth estimation functions

 ---
 Phase 3: Viewport Prediction Models (Days 4-6)

 Objective: Implement and train viewport predictor.

 3.1 Baselines (src/models/baseline.py)

 class NoPredictor:
     # Download all tiles at high quality (upper bound quality, lower bound efficiency)

 class LinearPredictor:
     # pos(t+Δt) = pos(t) + velocity(t) × Δt

 class ContentAgnosticLSTM:
     # LSTM using only orientation history (no saliency/motion)

 3.2 Main Model: Content-Aware LSTM (src/models/attention_predictor.py) ⭐

 Architecture:
 class ContentAwareLSTM(nn.Module):
     # 1. Feature Encoders
     - saliency_encoder: ResNet18 (pretrained) → 128-dim
     - motion_encoder: ResNet18 (pretrained) → 128-dim
     - orientation_fc: Linear(3) → 64-dim (yaw/pitch/roll)
     - velocity_fc: Linear(3) → 64-dim (angular velocities)
     - video_embedding: Embedding(10) → 64-dim (per-video patterns)

     # 2. Feature Fusion
     - fusion: Linear(128+128+64+64+64) → 256-dim

     # 3. Temporal Model
     - lstm: 2-layer LSTM(256, 256) with dropout=0.3

     # 4. Spatial Attention
     - attention: MultiheadAttention(256, heads=8)

     # 5. Multi-horizon Prediction
     - fc_1s: MLP(512 → 256 → 3)  # 1 second ahead
     - fc_2s: MLP(512 → 256 → 3)  # 2 seconds ahead
     - fc_3s: MLP(512 → 256 → 3)  # 3 seconds ahead

 Key Improvements over Prior Work:
 1. Content awareness: Uses pre-computed saliency/motion to focus on attractive regions
 2. Multi-horizon prediction: Predicts 1s/2s/3s ahead for adaptive prefetching
 3. Video-specific embeddings: Learns per-video viewing patterns
 4. Attention mechanism: Weighs important spatial/temporal features

 Training (src/models/trainer.py):
 - Loss: MSE(pred_1s) + 0.5*MSE(pred_2s) + 0.25*MSE(pred_3s)
 - Optimizer: Adam, lr=1e-3, cosine annealing
 - Batch size: 64 sequences
 - Sequence length: N=30 frames (1 second history)
 - Regularization: Dropout 0.3, gradient clipping
 - Early stopping on validation MSE

 Deliverables:
 - Trained model checkpoints in /outputs/models/
 - Training curves (loss, validation MSE)
 - Prediction accuracy per video (tile IoU, angular error)

 ---
 Phase 4: ABR Simulation (Days 7-9)

 Objective: Simulate streaming sessions with different strategies.

 4.1 Bandwidth Model (src/simulation/bandwidth_model.py)

 class BandwidthModel:
     def generate_trace(duration, profile='variable'):
         # Profiles:
         # - 'stable': 5 Mbps constant
         # - 'variable': 2-10 Mbps (Brownian motion)
         # - 'poor': 1-3 Mbps
         # Returns: bandwidth per segment (e.g., every 2s)

 4.2 Tile Selection (src/simulation/tile_selector.py) ⭐

 class TileSelector:
     def select_tiles(predicted_viewport, available_bandwidth, buffer_level):
         """
         Strategy:
         1. Predict viewport tiles for next 2 seconds
         2. Classify tiles:
            - Center (predicted viewport): HIGH quality
            - Periphery (adjacent): MEDIUM quality
            - Background (rest): LOW quality or skip
         3. Bandwidth allocation (priority order):
            - If buffer > 5s: Maximize quality (center first)
            - If buffer < 2s: Minimize stalls (any quality)
         4. Return: List[(tile_id, quality_level)]
         """

 Strategies to Compare:
 1. Full Download: All tiles at high quality (baseline)
 2. Uniform ABR: All tiles at same adaptive quality
 3. Predicted Tiling (Ours): Viewport prediction + quality allocation
 4. Oracle: Perfect future knowledge (upper bound)

 4.3 Session Simulator (src/simulation/session_simulator.py)

 class SessionSimulator:
     def simulate_session(video_name, user_id, predictor, bandwidth_trace):
         for frame in session:
             # 1. Predict viewport
             predicted_tiles = predictor.predict(orientation_history)

             # 2. Select tiles based on bandwidth
             tile_requests = tile_selector.select(predicted_tiles,
                                                   current_bandwidth,
                                                   buffer_level)

             # 3. Download tiles
             bytes_downloaded = sum(tile_size[tile] for tile in tile_requests)
             download_time = bytes_downloaded / current_bandwidth

             # 4. Update buffer
             buffer.consume(frame_duration)
             buffer.add(downloaded_tiles)

             # 5. Check quality at actual viewport
             actual_tiles = ground_truth_tiles[frame]
             viewport_quality = get_quality(actual_tiles, downloaded_tiles)

             # 6. Log metrics
             log(bandwidth=bytes_downloaded,
                 quality=viewport_quality,
                 rebuffer=buffer.stall_time,
                 prediction_iou=tile_overlap(predicted, actual))

 Deliverables:
 - Simulation logs for all 500 sessions × 4 strategies
 - CSV with per-frame metrics

 ---
 Phase 5: Evaluation (Days 10-11)

 Objective: Compute metrics and generate plots.

 5.1 Metrics (src/evaluation/metrics.py) ⭐

 class Metrics:
     def bandwidth_savings(strategy_bandwidth, baseline_bandwidth):
         # (1 - strategy/baseline) × 100%
         # Goal: >50%

     def viewport_quality(rendered_quality, full_quality):
         # PSNR/SSIM of viewport tiles
         # If no actual video: use quality scores (high=1, medium=0.7, low=0.3)
         # Goal: PSNR >35 dB or avg quality >0.8

     def stall_ratio(total_stall_time, total_playback_time):
         # % time rebuffering
         # Goal: <2%

     def prediction_hit_rate(predicted_tiles, actual_tiles):
         # IoU across all frames
         # Goal: >70% at 1s horizon

 5.2 Visualizations (src/evaluation/visualizer.py)

 1. Main Results:
   - Bar chart: Bandwidth savings per strategy
   - Scatter: Bandwidth vs. QoE tradeoff
   - Line: Prediction IoU vs. time horizon (1s/2s/3s)
 2. Per-Video Breakdown:
   - Table: Metrics per video (identify which content benefits most)
 3. Session Examples:
   - Heatmap: Predicted vs. actual viewport over time
   - Timeline: Bandwidth, buffer level, quality over session

 Deliverables:
 - Publication-ready plots in /outputs/plots/
 - Summary CSV with aggregate metrics
 - Ablation study results (e.g., without content features)

 ---
 Phase 6: Documentation (Days 12-14)

 1. README.md:
   - Setup instructions
   - How to run scripts 01-05
   - Expected results
 2. Results Summary:
   - Key findings (bandwidth savings, QoE, prediction accuracy)
   - Comparison table vs. baselines
 3. Optional: Paper draft (methods + results sections)

 ---
 Critical Design Decisions

 1. Tile Encoding: Estimation vs. Actual

 Recommendation: Use bandwidth estimation initially
 - Rationale: 10× faster, sufficient for proof-of-concept
 - Estimation: High=50KB, Medium=25KB, Low=10KB per tile
 - Upgrade path: Encode 2-3 videos later for validation

 2. Prediction Horizon

 Recommendation: Multi-horizon (1s, 2s, 3s)
 - Rationale: 1s for immediate fetch (high confidence), 2-3s for prefetch (lower quality acceptable)
 - Implementation: Model outputs 3 predictions, tile selector uses all

 3. Content Features

 Recommendation: Use saliency + motion as input
 - Rationale: Pre-computed, easy to extract, differentiates from position-only baselines
 - Implementation: Extract 224×224 patches around viewport, encode with ResNet-18

 4. Training Split

 Recommendation: Leave-one-video-out cross-validation
 - Rationale: Tests generalization to new content (research goal)
 - Implementation: 10-fold CV (train on 9 videos, test on 1)

 5. Simulation Complexity

 Recommendation: Start simple, add realism iteratively
 - Initial: Fixed buffer (5s), synthetic bandwidth (Brownian motion)
 - Later: Variable buffer, trace-based bandwidth

 ---
 Development Workflow

 Week 1 (Fast Track):

 - Day 1: Setup, data_loader.py, viewport.py, exploration notebook
 - Day 2: dataset_builder.py, preprocess all data
 - Day 3: baseline.py, lstm_predictor.py (basic version)
 - Day 4: Train baselines (linear, LSTM)
 - Day 5: attention_predictor.py, train full model
 - Day 6: tile_selector.py, bandwidth_model.py
 - Day 7: session_simulator.py, run 500 sessions

 Week 2 (Refinement):

 - Day 8: metrics.py, compute all metrics
 - Day 9: visualizer.py, generate plots
 - Day 10: Ablation studies (remove content features, attention)
 - Day 11: Hyperparameter tuning
 - Day 12-13: Documentation, README, notebooks
 - Day 14: Paper draft (optional)

 ---
 Risk Mitigation

 | Risk                     | Mitigation                                               |
 |--------------------------|----------------------------------------------------------|
 | Tile encoding too slow   | Use bandwidth estimation instead of actual encoding      |
 | Poor prediction accuracy | Strong baselines prove value even if model underperforms |
 | Missing original videos  | Use quality scores instead of PSNR/SSIM                  |
 | Simulation too complex   | Start with simplified buffer/bandwidth models            |
 | Time pressure            | Focus on 2-3 videos, 100 sessions for initial results    |

 ---
 Success Criteria

 Minimum Viable Results (7 days):
 - ✅ Working data pipeline
 - ✅ Trained LSTM predictor (>60% tile IoU at 1s)
 - ✅ Simulation showing >30% bandwidth savings vs. full download
 - ✅ Basic plots (bandwidth, quality, prediction accuracy)

 Full Results (14 days):
 - ✅ All 500 sessions simulated
 - ✅ >50% bandwidth savings with >0.8 viewport quality
 - ✅ <2% stall ratio under variable bandwidth
 - ✅ Ablation studies and comparisons
 - ✅ Publication-ready figures and draft

 ---
 Most Critical Files (Build These First)

 1. src/preprocessing/data_loader.py - Foundation for all data access
 2. src/utils/viewport.py - Core geometry for orientation→tiles
 3. src/models/lstm_predictor.py - Proves prediction concept
 4. src/simulation/tile_selector.py - Core ABR innovation
 5. src/evaluation/metrics.py - Defines success

 Day 1 Priority: Build data_loader.py and viewport.py, validate with exploration notebook before proceeding.
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
