"""
Static Digital Twin Visualization Generator
------------------------------------------
Generates publication-quality trajectory visualizations
matching the reference image style (dotted green/red paths)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import torch
import os
import sys
from pathlib import Path

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import SimpleSLSTM, ImprovedTrajectoryTransformer


# ==================== Configuration ====================
class VizConfig:
    MODEL_PATH = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    MODEL_TYPE = "slstm"
    CSV_PATH = "./simulated/data/sumo_direction1.csv"
    OUTPUT_DIR = "./dt_static_visualizations"
    
    OBS_LEN = 20
    PRED_LEN = 20
    DISPLAY_LEN = 12  # Show 3 seconds only
    K_NEIGHBORS = 8
    
    # Visualization style
    ROAD_WIDTH = 30
    LANE_WIDTH = 10
    LINE_SPACING = 2  # Dotted effect
    PRED_COLOR = '#2ecc71'  # Green
    TRUE_COLOR = '#e74c3c'  # Red
    LINE_WIDTH = 2.8
    DPI = 250


# ==================== Device Setup ====================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ==================== Model Loading ====================
def load_model(config):
    """Load trained model"""
    if config.MODEL_TYPE == "slstm":
        model = SimpleSLSTM(pred_len=config.PRED_LEN)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=config.PRED_LEN)
    
    state_dict = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


# ==================== Visualization Generator ====================
def generate_dt_visualizations(config, num_samples=10):
    """Generate Digital Twin style visualizations"""
    
    print("\n" + "="*70)
    print("DIGITAL TWIN STATIC VISUALIZATION GENERATOR")
    print("="*70)
    print(f"Model: {config.MODEL_TYPE.upper()}")
    print(f"CSV: {config.CSV_PATH}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"Samples: {num_samples}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = load_model(config)
    print(f"✓ Model loaded (k={model.k} neighbors)\n")
    
    # Load CSV data
    print("Loading trajectory data...")
    df = pd.read_csv(config.CSV_PATH)
    print(f"✓ Loaded {len(df)} records from {len(df['id'].unique())} vehicles\n")
    
    # Create output directory
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Group by vehicle
    vehicle_groups = df.groupby('id')
    
    print(f"Generating {num_samples} visualizations...\n")
    
    sample_count = 0
    processed = 0
    
    for vid, group in vehicle_groups:
        if sample_count >= num_samples:
            break
        
        processed += 1
        
        # Sort by frame
        group = group.sort_values('frame')
        
        # Need enough data
        if len(group) < config.OBS_LEN + config.PRED_LEN:
            continue
        
        # Extract observation and ground truth
        obs_data = group.iloc[:config.OBS_LEN]
        gt_data = group.iloc[config.OBS_LEN:config.OBS_LEN + config.PRED_LEN]
        
        # Prepare observation tensor [obs_len, 7]
        obs = np.zeros((config.OBS_LEN, 7))
        obs[:, 0] = obs_data['x'].values
        obs[:, 1] = 0
        obs[:, 2] = obs_data['v'].values
        obs[:, 3] = 0
        obs[:, 4] = obs_data['acc'].values if 'acc' in obs_data.columns else 0
        obs[:, 5] = 0
        obs[:, 6] = obs_data['lane_index'].values
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Dummy neighbor/lane data
        nd = torch.zeros(1, model.k, config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        try:
            # Make prediction
            with torch.no_grad():
                if hasattr(model, "multi_att"):  # Transformer
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = model(obs_tensor, nd, ns, lane)
            
            pred_np = pred.cpu().numpy()[0]
            
            # Generate visualization
            success = create_single_visualization(
                vid, obs, pred_np, gt_data, config
            )
            
            if success:
                sample_count += 1
                print(f"  ✓ [{sample_count}/{num_samples}] Vehicle {vid}")
            
        except Exception as e:
            print(f"  ✗ Vehicle {vid} failed: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {sample_count} visualizations")
    print(f"  Processed: {processed} vehicles")
    print(f"  Success rate: {sample_count/processed*100:.1f}%")
    print(f"  Output: {config.OUTPUT_DIR}/")
    print("="*70 + "\n")


def create_single_visualization(vid, obs, pred, gt_data, config):
    """Create single Digital Twin visualization matching reference image"""
    
    try:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # === Road Background ===
        road_width = config.ROAD_WIDTH
        lane_width = config.LANE_WIDTH
        
        # Get x-range
        x_min = obs[0, 0] - 50
        x_max = obs[-1, 0] + gt_data['x'].iloc[-1] - obs[-1, 0] + 100
        
        # Road outline (gray background)
        ax.add_patch(patches.Rectangle(
            (x_min, -road_width / 2),
            x_max - x_min,
            road_width,
            facecolor='#d0d0d0',
            edgecolor='black',
            linewidth=1.5,
            zorder=0
        ))
        
        # Lane markings (white dashed)
        for i in range(1, 3):
            lane_y = -road_width / 2 + i * lane_width
            ax.plot([x_min, x_max], [lane_y, lane_y], 
                   color='white', linewidth=2, linestyle='--', 
                   alpha=0.7, zorder=1)
        
        # === Get vehicle lane ===
        lane_id = int(obs[0, 6])
        lane_y = -road_width / 2 + (lane_id + 0.5) * lane_width
        
        # === Ground Truth Trajectory (RED DOTTED) ===
        gt_x = np.concatenate([[obs[-1, 0]], gt_data['x'].values])
        gt_y = np.full_like(gt_x, lane_y)
        
        # Use display length
        display_len = min(config.DISPLAY_LEN, len(gt_x) - 1)
        
        # Create dotted effect
        gt_points_x = [gt_x[0]]
        gt_points_y = [gt_y[0]]
        for i in range(0, display_len, config.LINE_SPACING):
            gt_points_x.append(gt_x[i + 1])
            gt_points_y.append(gt_y[i + 1])
        
        ax.plot(gt_points_x, gt_points_y, 
               color=config.TRUE_COLOR, linestyle='--', 
               linewidth=config.LINE_WIDTH, marker='o', markersize=3,
               label='True trajectory', zorder=5)
        
        # === Predicted Trajectory (GREEN DOTTED) ===
        pred_cumsum_x = np.cumsum(np.concatenate([[0], pred[:display_len, 0]]))
        pred_cumsum_y = np.cumsum(np.concatenate([[0], pred[:display_len, 1]]))
        
        pred_x = obs[-1, 0] + pred_cumsum_x
        pred_y = lane_y + pred_cumsum_y
        
        # Create dotted effect
        pred_points_x = [obs[-1, 0]]
        pred_points_y = [lane_y]
        for i in range(0, len(pred_cumsum_x), config.LINE_SPACING):
            pred_points_x.append(pred_x[i])
            pred_points_y.append(pred_y[i])
        
        ax.plot(pred_points_x, pred_points_y, 
               color=config.PRED_COLOR, linestyle='--', 
               linewidth=config.LINE_WIDTH, marker='o', markersize=3,
               label='Predicted trajectory', zorder=6)
        
        # === Vehicle Position (Blue Rectangle) ===
        vehicle_length = 4.5
        vehicle_width = 2.0
        ax.add_patch(patches.Rectangle(
            (obs[-1, 0] - vehicle_length/2, lane_y - vehicle_width/2),
            vehicle_length,
            vehicle_width,
            facecolor='#3498db',
            edgecolor='black',
            linewidth=1.5,
            zorder=10
        ))
        
        # === Styling ===
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-road_width/2 - 3, road_width/2 + 3)
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_title(f'Digital Twin Trajectory Prediction - Vehicle {vid}', 
                    fontsize=16, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper left', fontsize=13, framealpha=0.9)
        
        # Grid
        ax.grid(False)
        
        # Clean layout
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, f'dt_viz_{vid}.png')
        plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"    Error creating visualization: {e}")
        plt.close()
        return False


# ==================== Batch Comparison Visualization ====================
def create_comparison_grid(config, num_samples=6):
    """Create grid of multiple vehicle predictions"""
    
    print("\nGenerating comparison grid...")
    
    model = load_model(config)
    df = pd.read_csv(config.CSV_PATH)
    vehicle_groups = df.groupby('id')
    
    # Collect predictions
    predictions = []
    for vid, group in vehicle_groups:
        if len(predictions) >= num_samples:
            break
        
        group = group.sort_values('frame')
        if len(group) < config.OBS_LEN + config.PRED_LEN:
            continue
        
        obs_data = group.iloc[:config.OBS_LEN]
        gt_data = group.iloc[config.OBS_LEN:config.OBS_LEN + config.PRED_LEN]
        
        obs = np.zeros((config.OBS_LEN, 7))
        obs[:, 0] = obs_data['x'].values
        obs[:, 2] = obs_data['v'].values
        obs[:, 4] = obs_data['acc'].values if 'acc' in obs_data.columns else 0
        obs[:, 6] = obs_data['lane_index'].values
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        nd = torch.zeros(1, model.k, config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        try:
            with torch.no_grad():
                if hasattr(model, "multi_att"):
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = model(obs_tensor, nd, ns, lane)
            
            pred_np = pred.cpu().numpy()[0]
            predictions.append({
                'vid': vid,
                'obs': obs,
                'pred': pred_np,
                'gt': gt_data
            })
        except:
            continue
    
    # Create grid
    n = len(predictions)
    if n == 0:
        print("  ✗ No valid predictions found")
        return
    
    cols = 2
    rows = (n + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Digital Twin Predictions - Multiple Vehicles', 
                fontsize=18, fontweight='bold')
    
    for idx, data in enumerate(predictions):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        vid = data['vid']
        obs = data['obs']
        pred = data['pred']
        gt = data['gt']
        
        # Simple line plot
        display_len = min(config.DISPLAY_LEN, config.PRED_LEN)
        
        # Ground truth
        gt_x = gt['x'].values[:display_len]
        ax.plot(range(len(gt_x)), gt_x - obs[-1, 0], 
               color=config.TRUE_COLOR, linestyle='--', 
               linewidth=2, label='True', marker='o', markersize=3)
        
        # Prediction
        pred_cumsum = np.cumsum(pred[:display_len, 0])
        ax.plot(range(len(pred_cumsum)), pred_cumsum, 
               color=config.PRED_COLOR, linestyle='--', 
               linewidth=2, label='Predicted', marker='o', markersize=3)
        
        ax.set_xlabel('Frame', fontweight='bold')
        ax.set_ylabel('Relative X (m)', fontweight='bold')
        ax.set_title(f'Vehicle {vid}', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(config.OUTPUT_DIR, 'comparison_grid.png')
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f" Saved comparison grid: {output_path}")


# ==================== Main ====================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DT visualizations")
    parser.add_argument("--csv_path", type=str, 
                       default="./simulated/data/sumo_direction1.csv")
    parser.add_argument("--model_path", type=str,
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], 
                       default="slstm")
    parser.add_argument("--output_dir", type=str, 
                       default="./dt_static_visualizations")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--comparison_grid", action="store_true",
                       help="Also generate comparison grid")
    
    args = parser.parse_args()
    
    # Update config
    config = VizConfig()
    config.CSV_PATH = args.csv_path
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.OUTPUT_DIR = args.output_dir
    
    # Check files exist
    if not os.path.exists(config.CSV_PATH):
        print(f" CSV file not found: {config.CSV_PATH}")
        print("Run main.py or visualize_predictions.py first to generate data!")
        return
    
    if not os.path.exists(config.MODEL_PATH):
        print(f" Model not found: {config.MODEL_PATH}")
        return
    
    # Generate visualizations
    generate_dt_visualizations(config, args.num_samples)
    
    # Optional comparison grid
    if args.comparison_grid:
        create_comparison_grid(config, num_samples=6)


if __name__ == "__main__":
    main()