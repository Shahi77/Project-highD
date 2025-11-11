"""
Enhanced Static Trajectory Visualizations - Research Paper Quality
------------------------------------------------------------------
Creates publication-ready visualizations with proper trajectory separation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import torch
import sys

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import SimpleSLSTM, ImprovedTrajectoryTransformer

# Configuration
MODEL_PATH = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
MODEL_TYPE = "slstm"
CSV_PATH = "./simulated/data/sumo_direction1.csv"
SAVE_DIR = "./paper_quality_viz"
NUM_SAMPLES = 10
OBS_LEN = 20
PRED_LEN = 25

device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")


def load_model(model_path, model_type="slstm"):
    """Load trained model"""
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=PRED_LEN)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=PRED_LEN)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def draw_vehicle_box(ax, x, y, width=4.5, height=2.0, color='#3498db', 
                     linewidth=1.5, alpha=1.0, label=None):
    """Draw a vehicle as a rounded rectangle"""
    vehicle = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor='black',
        linewidth=linewidth,
        alpha=alpha,
        zorder=15,
        label=label
    )
    ax.add_patch(vehicle)
    return vehicle


def create_highway_scene(ax, x_start, x_end, num_lanes=3, add_context_vehicles=True):
    """
    Create realistic highway scene with proper lane markings
    """
    lane_width = 3.5
    total_width = num_lanes * lane_width
    
    # Main road surface (dark gray)
    road = Rectangle(
        (x_start - 30, -total_width/2 - 0.5),
        x_end - x_start + 60,
        total_width + 1,
        facecolor='#3d3d3d',
        edgecolor='#1a1a1a',
        linewidth=2.5,
        zorder=1
    )
    ax.add_patch(road)
    
    # Lane dividing lines (white dashed)
    for i in range(1, num_lanes):
        y_pos = -total_width/2 + i * lane_width
        x = x_start - 30
        dash_len, gap_len = 6, 4
        
        while x < x_end + 30:
            ax.plot([x, x + dash_len], [y_pos, y_pos],
                   'w-', linewidth=1.8, alpha=0.85, zorder=2)
            x += dash_len + gap_len
    
    # Road edge lines (solid white)
    for y_edge in [-total_width/2, total_width/2]:
        ax.plot([x_start - 30, x_end + 30], [y_edge, y_edge],
               'w-', linewidth=2.5, alpha=0.9, zorder=2)
    
    # Add context vehicles if requested
    if add_context_vehicles:
        # Add a few gray vehicles in other lanes for context
        context_positions = [
            (x_start + 40, 0, '#95a5a6'),  # Lane 1
            (x_start + 120, 1, '#7f8c8d'),  # Lane 2
            (x_end - 80, 2, '#95a5a6'),  # Lane 3
        ]
        
        for x_pos, lane, color in context_positions:
            y_pos = -total_width/2 + (lane + 0.5) * lane_width
            draw_vehicle_box(ax, x_pos, y_pos, color=color, alpha=0.6)
    
    return lane_width


def visualize_trajectory_separated(obs, pred, gt, vehicle_id, lane_id, 
                                   save_path, pred_display_len=15):
    """
    Create visualization with CLEAR SEPARATION between trajectories
    
    Key improvements:
    - Lateral offset between prediction and ground truth
    - Shorter trajectory lengths for clarity
    - Dotted/dashed styles
    - Proper markers and colors
    """
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Calculate plot extent
    all_x = np.concatenate([obs[:, 0], gt[:pred_display_len, 0]])
    x_min, x_max = all_x.min() - 40, all_x.max() + 40
    
    # Draw highway background
    lane_width = create_highway_scene(ax, x_min, x_max, num_lanes=3)
    base_lane_y = -lane_width * 1.5 + (lane_id + 0.5) * lane_width
    
    # Define lateral offsets for trajectory separation
    offset_gt = 0.0  # Ground truth at center
    offset_pred = 0.8  # Prediction offset by 0.8m
    offset_obs = 0.0  # Observation at center
    
    # Limit display length
    pred_display_len = min(pred_display_len, len(pred))
    gt_display_len = min(pred_display_len, len(gt))
    obs_display_len = min(12, len(obs))  # Show last 12 observation points
    
    # --- 1. OBSERVATION (Yellow, solid line with circles) ---
    obs_x = obs[-obs_display_len:, 0]
    obs_y = np.full_like(obs_x, base_lane_y + offset_obs)
    
    ax.plot(obs_x, obs_y, 'o-', 
           color='#f39c12', linewidth=3, markersize=6,
           label='Observation (past)', zorder=8,
           markeredgecolor='#c87f0a', markeredgewidth=1)
    
    # --- 2. GROUND TRUTH (Red, dashed line with circles) ---
    gt_x = gt[:gt_display_len, 0]
    gt_y = np.full_like(gt_x, base_lane_y + offset_gt)
    
    # Sample every 2nd point for dotted effect
    gt_x_sampled = gt_x[::2]
    gt_y_sampled = gt_y[::2]
    
    ax.plot(gt_x_sampled, gt_y_sampled, 'o--',
           color='#e74c3c', linewidth=2.5, markersize=5,
           label='Ground truth (actual)', zorder=6,
           markeredgecolor='#c0392b', markeredgewidth=0.8,
           dashes=(6, 3))
    
    # --- 3. PREDICTION (Green, solid line with squares) ---
    # Convert relative predictions to absolute positions
    pred_x = obs[-1, 0] + np.cumsum(np.concatenate([[0], pred[:pred_display_len, 0]]))
    pred_y = base_lane_y + offset_pred + np.cumsum(np.concatenate([[0], pred[:pred_display_len, 1]]))
    
    # Sample every 2nd point
    pred_x_sampled = pred_x[::2]
    pred_y_sampled = pred_y[::2]
    
    ax.plot(pred_x_sampled, pred_y_sampled, 's-',
           color='#27ae60', linewidth=2.5, markersize=5,
           label='Prediction (model)', zorder=7,
           markeredgecolor='#1e8449', markeredgewidth=0.8)
    
    # --- 4. DRAW VEHICLES ---
    # Main vehicle at current position (Blue)
    draw_vehicle_box(ax, obs[-1, 0], base_lane_y, 
                    color='#3498db', width=4.8, height=2.2)
    
    # Predicted end position (Green vehicle)
    draw_vehicle_box(ax, pred_x[-1], pred_y[-1], 
                    color='#27ae60', width=4.2, height=2.0, alpha=0.85)
    
    # Ground truth end position (Red vehicle)
    draw_vehicle_box(ax, gt_x[-1], gt_y[-1], 
                    color='#e74c3c', width=4.2, height=2.0, alpha=0.85)
    
    # --- Add trajectory direction arrows ---
    # Arrow for prediction
    if len(pred_x_sampled) > 2:
        mid_idx = len(pred_x_sampled) // 2
        dx = pred_x_sampled[mid_idx + 1] - pred_x_sampled[mid_idx]
        dy = pred_y_sampled[mid_idx + 1] - pred_y_sampled[mid_idx]
        ax.arrow(pred_x_sampled[mid_idx], pred_y_sampled[mid_idx], 
                dx * 0.3, dy * 0.3,
                head_width=1.2, head_length=2, fc='#27ae60', ec='#27ae60',
                zorder=9, linewidth=1.5)
    
    # --- Styling ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-lane_width * 1.5 - 2, lane_width * 1.5 + 2)
    ax.set_aspect('equal')
    
    ax.set_xlabel('Longitudinal Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Lateral Position (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'Trajectory Prediction Comparison - Vehicle {vehicle_id}', 
                fontsize=15, fontweight='bold', pad=12)
    
    # Enhanced legend
    legend = ax.legend(loc='upper left', fontsize=11.5, 
                      framealpha=0.95, edgecolor='black', 
                      fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Grid with low opacity
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, color='gray')
    ax.set_facecolor('#e8e8e8')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Generate high-quality trajectory visualizations"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("="*75)
    print(" Publication-Quality Trajectory Prediction Visualizations")
    print("="*75)
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Dataset: {CSV_PATH}")
    print(f"Output: {SAVE_DIR}")
    print("="*75 + "\n")
    
    # Load model
    print("Loading model...")
    model = load_model(MODEL_PATH, MODEL_TYPE)
    print(f" Model loaded (k={model.k} neighbors)\n")
    
    # Load data
    print("Loading trajectory data...")
    df = pd.read_csv(CSV_PATH)
    vehicle_groups = df.groupby('id')
    print(f" Found {len(vehicle_groups)} vehicles\n")
    
    print(f"Generating {NUM_SAMPLES} visualizations...\n")
    
    viz_count = 0
    
    for vid, group in vehicle_groups:
        if viz_count >= NUM_SAMPLES:
            break
        
        group = group.sort_values('frame').reset_index(drop=True)
        
        if len(group) < OBS_LEN + PRED_LEN:
            continue
        
        # Extract data
        obs_data = group.iloc[:OBS_LEN]
        gt_data = group.iloc[OBS_LEN:OBS_LEN + PRED_LEN]
        lane_id = int(obs_data.iloc[0]['lane_index'])
        
        # Prepare observation
        obs = np.zeros((OBS_LEN, 7))
        obs[:, 0] = obs_data['x'].values
        obs[:, 2] = obs_data['v'].values
        obs[:, 4] = obs_data['acc'].values if 'acc' in obs_data.columns else 0
        obs[:, 6] = obs_data['lane_index'].values
        
        # Ground truth
        gt = np.zeros((PRED_LEN, 2))
        gt[:, 0] = gt_data['x'].values
        
        # Make prediction
        try:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            nd = torch.zeros(1, model.k, OBS_LEN, 7).to(device)
            ns = torch.zeros(1, model.k, 2).to(device)
            lane = torch.zeros(1, 3).to(device)
            
            with torch.no_grad():
                if hasattr(model, "multi_att"):
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = model(obs_tensor, nd, ns, lane)
            
            pred_np = pred.cpu().numpy()[0]
            
            # Create visualization with trajectory separation
            save_path = os.path.join(SAVE_DIR, f'vehicle_{vid}_trajectory.png')
            visualize_trajectory_separated(obs, pred_np, gt, vid, lane_id, 
                                          save_path, pred_display_len=15)
            
            viz_count += 1
            print(f" [{viz_count:02d}/{NUM_SAMPLES}] Vehicle {vid:15s}")
            
        except Exception as e:
            print(f"  Error with vehicle {vid}: {str(e)[:50]}")
            continue
    
    print("\n" + "="*75)
    print(f" Successfully generated {viz_count} visualizations")
    print(f" Location: {os.path.abspath(SAVE_DIR)}/")
    print("="*75)
    print("\n Visualization Key:")
    print("  • Yellow line + circles  = Observation (vehicle's past)")
    print("  • Red dashed + circles   = Ground truth (actual future)")
    print("  • Green line + squares   = Prediction (model forecast)")
    print("  • Blue vehicle           = Current position")
    print("  • Red/Green vehicles     = End positions (actual vs predicted)")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()