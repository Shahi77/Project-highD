"""
Fixed Digital Twin SUMO Simulation - WORKING METRICS COLLECTION
--------------------------------------------------------------
Based on the paper's methodology:
- Observation time T_hist = 20 frames (5 seconds at 4 Hz)
- Prediction time T_fut = 20 frames (5 seconds at 4 Hz)
- Collect metrics immediately when we have enough ground truth data
"""

import os
import sys
import random
import numpy as np
import torch
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from utils import check_sumo_env, start_sumo, running

check_sumo_env()
import traci
from traci import constants as tc


@dataclass
class DTConfig:
    MODEL_TYPE: str = "slstm"
    MODEL_PATH: str = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    OBS_LEN: int = 20  # 5 seconds at 4 Hz
    PRED_LEN: int = 20  # 5 seconds at 4 Hz
    K_NEIGHBORS: int = 8
    PREDICTION_INTERVAL_MS: int = 500  # Predict every 0.5s
    
    # Metrics collection - SIMPLIFIED
    MIN_GT_FRAMES: int = 8  # Minimum 2 seconds of ground truth
    MAX_GT_WAIT_STEPS: int = 100  # Maximum 25 seconds to wait
    
    # Visualization
    DRAW_PREDICTIONS: bool = True
    PRED_DISPLAY_LEN: int = 8
    DASH_LENGTH: float = 0.3
    DASH_GAP: float = 0.15
    LINE_WIDTH: float = 0.25
    LATERAL_OFFSET: float = 0.35
    
    # Colors
    PRED_LINE_COLOR: Tuple[int, int, int, int] = (0, 255, 0, 255)
    TRUE_LINE_COLOR: Tuple[int, int, int, int] = (255, 0, 0, 255)
    VEHICLE_COLOR: Tuple[int, int, int, int] = (255, 255, 0, 255)
    
    GUI: bool = True
    TOTAL_TIME: int = 4000
    START_STEP: int = 100
    USE_DT_PREDICTION: bool = True
    METRICS_OUTPUT: str = "./dt_results/dt_metrics_working.json"


config = DTConfig()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class PredictionMetrics:
    vehicle_id: str
    prediction_step: int
    ade: float
    fde: float
    inference_latency_ms: float
    e2e_latency_ms: float
    num_gt_frames: int
    prediction_timestamp: float


class SimpleMetricsCollector:
    def __init__(self, config: DTConfig):
        self.config = config
        self.metrics: List[PredictionMetrics] = []
        self.start_time = time.time()
        
    def add_metric(self, vehicle_id: str, prediction_step: int,
                   pred_traj: np.ndarray, true_traj: np.ndarray,
                   inference_ms: float, e2e_ms: float,
                   timestamp: float):
        """Add a single prediction metric"""
        # Calculate ADE and FDE
        displacements = np.linalg.norm(pred_traj - true_traj, axis=1)
        ade = float(np.mean(displacements))
        fde = float(displacements[-1])
        
        metric = PredictionMetrics(
            vehicle_id=vehicle_id,
            prediction_step=prediction_step,
            ade=ade,
            fde=fde,
            inference_latency_ms=inference_ms,
            e2e_latency_ms=e2e_ms,
            num_gt_frames=len(true_traj),
            prediction_timestamp=timestamp
        )
        
        self.metrics.append(metric)
    
    def get_summary(self) -> Dict:
        """Calculate summary statistics"""
        if not self.metrics:
            return {}
        
        ades = [m.ade for m in self.metrics]
        fdes = [m.fde for m in self.metrics]
        inference_times = [m.inference_latency_ms for m in self.metrics]
        e2e_times = [m.e2e_latency_ms for m in self.metrics]
        
        return {
            "trajectory_accuracy": {
                "ADE_mean": float(np.mean(ades)),
                "ADE_std": float(np.std(ades)),
                "ADE_median": float(np.median(ades)),
                "ADE_min": float(np.min(ades)),
                "ADE_max": float(np.max(ades)),
                "FDE_mean": float(np.mean(fdes)),
                "FDE_std": float(np.std(fdes)),
                "FDE_median": float(np.median(fdes)),
                "FDE_min": float(np.min(fdes)),
                "FDE_max": float(np.max(fdes)),
            },
            "latency_metrics": {
                "inference_mean_ms": float(np.mean(inference_times)),
                "inference_std_ms": float(np.std(inference_times)),
                "inference_p95_ms": float(np.percentile(inference_times, 95)),
                "inference_p99_ms": float(np.percentile(inference_times, 99)),
                "e2e_mean_ms": float(np.mean(e2e_times)),
                "e2e_std_ms": float(np.std(e2e_times)),
            },
            "performance_summary": {
                "total_predictions": len(self.metrics),
                "unique_vehicles": len(set(m.vehicle_id for m in self.metrics)),
                "avg_gt_frames": float(np.mean([m.num_gt_frames for m in self.metrics])),
                "simulation_time_s": time.time() - self.start_time,
            }
        }
    
    def save_results(self, filepath: str):
        """Save metrics to file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        summary = self.get_summary()
        summary["raw_predictions"] = [asdict(m) for m in self.metrics[-100:]]  # Last 100 samples
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved {len(self.metrics)} metrics to {filepath}")


def load_model(model_path: str, model_type: str, pred_len: int):
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Loaded {model_type.upper()} model")
    return model


class TrajectoryTracker:
    """Track historical trajectories for observations"""
    def __init__(self, obs_len: int = 20):
        self.obs_len = obs_len
        self.trajectories = defaultdict(lambda: deque(maxlen=obs_len))
        
    def update(self, vehicle_id: str, position: Tuple[float, float], 
               velocity: float, acceleration: float, lane_id: int):
        self.trajectories[vehicle_id].append({
            'x': position[0], 'y': position[1],
            'vx': velocity, 'vy': 0.0,
            'ax': acceleration, 'ay': 0.0,
            'lane_id': lane_id
        })
    
    def get_observation(self, vehicle_id: str) -> Optional[np.ndarray]:
        if vehicle_id not in self.trajectories:
            return None
        
        traj = list(self.trajectories[vehicle_id])
        if len(traj) < self.obs_len:
            return None
        
        obs = np.zeros((self.obs_len, 7))
        for i, frame in enumerate(traj):
            obs[i] = [frame['x'], frame['y'], frame['vx'], frame['vy'],
                     frame['ax'], frame['ay'], frame['lane_id']]
        
        return obs
    
    def has_enough_history(self, vehicle_id: str) -> bool:
        return (vehicle_id in self.trajectories and 
                len(self.trajectories[vehicle_id]) >= self.obs_len)


class PredictionRecord:
    """Store a single prediction and collect its ground truth"""
    def __init__(self, vehicle_id: str, prediction_step: int, 
                 prediction: np.ndarray, start_pos: Tuple[float, float],
                 inference_ms: float, e2e_ms: float, timestamp: float):
        self.vehicle_id = vehicle_id
        self.prediction_step = prediction_step
        self.prediction = prediction  # Relative coordinates
        self.start_pos = start_pos
        self.inference_ms = inference_ms
        self.e2e_ms = e2e_ms
        self.timestamp = timestamp
        
        # Ground truth collection
        self.ground_truth: List[Tuple[float, float]] = []
        self.steps_waiting = 0
        self.completed = False
    
    def add_ground_truth_frame(self, position: Tuple[float, float]):
        """Add a ground truth position (absolute coordinates)"""
        if not self.completed:
            self.ground_truth.append(position)
    
    def can_evaluate(self, min_frames: int) -> bool:
        """Check if we have enough ground truth to evaluate"""
        return len(self.ground_truth) >= min_frames
    
    def should_timeout(self, max_steps: int) -> bool:
        """Check if we've waited too long"""
        self.steps_waiting += 1
        return self.steps_waiting > max_steps
    
    def get_metrics(self, max_len: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get prediction and ground truth arrays for evaluation"""
        if len(self.ground_truth) == 0:
            return None
        
        # Convert ground truth to relative coordinates
        gt_relative = np.zeros((len(self.ground_truth), 2))
        for i, pos in enumerate(self.ground_truth):
            gt_relative[i, 0] = pos[0] - self.start_pos[0]
            gt_relative[i, 1] = pos[1] - self.start_pos[1]
        
        # Use the minimum length available
        use_len = min(len(self.prediction), len(gt_relative), max_len)
        
        pred_traj = self.prediction[:use_len]
        true_traj = gt_relative[:use_len]
        
        return pred_traj, true_traj


class DigitalTwinPredictor:
    def __init__(self, model, tracker: TrajectoryTracker, 
                 metrics: SimpleMetricsCollector, config: DTConfig):
        self.model = model
        self.tracker = tracker
        self.metrics = metrics
        self.config = config
        
        # Prediction tracking
        self.active_predictions = {}  # For visualization
        self.prediction_records: Dict[str, PredictionRecord] = {}  # For metrics
        self.last_prediction_time = {}
        
        # Visualization
        self.drawn_objects = set()
        
        # Statistics
        self.total_predictions_made = 0
        self.total_metrics_collected = 0
        self.failed_collections = 0
    
    def should_predict(self, vehicle_id: str, current_time: float) -> bool:
        """Check if enough time has passed since last prediction"""
        if vehicle_id not in self.last_prediction_time:
            return True
        
        elapsed_ms = (current_time - self.last_prediction_time[vehicle_id]) * 1000
        return elapsed_ms >= self.config.PREDICTION_INTERVAL_MS
    
    def make_prediction(self, vehicle_id: str, current_time: float, current_step: int):
        """Make a prediction for a vehicle"""
        # Get observation
        obs = self.tracker.get_observation(vehicle_id)
        if obs is None:
            return False
        
        # Get current position
        try:
            start_pos = traci.vehicle.getPosition(vehicle_id)
        except:
            return False
        
        # Prepare input tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        nd = torch.zeros(1, self.model.k, self.config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, self.model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        # Run inference
        e2e_start = time.time()
        
        try:
            inference_start = time.time()
            with torch.no_grad():
                if hasattr(self.model, "multi_att"):
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = self.model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = self.model(obs_tensor, nd, ns, lane)
            
            inference_ms = (time.time() - inference_start) * 1000
            e2e_ms = (time.time() - e2e_start) * 1000
            
            pred_np = pred.cpu().numpy()[0]  # Shape: (PRED_LEN, 2)
            
            # Create prediction record for metrics collection
            record = PredictionRecord(
                vehicle_id=vehicle_id,
                prediction_step=current_step,
                prediction=pred_np,
                start_pos=start_pos,
                inference_ms=inference_ms,
                e2e_ms=e2e_ms,
                timestamp=current_time
            )
            
            # Store with unique key
            record_key = f"{vehicle_id}_{current_step}"
            self.prediction_records[record_key] = record
            
            # Store for visualization
            self.active_predictions[vehicle_id] = {
                'prediction': pred_np,
                'start_pos': start_pos,
                'timestamp': current_time
            }
            
            self.last_prediction_time[vehicle_id] = current_time
            self.total_predictions_made += 1
            
            return True
            
        except Exception as e:
            print(f"Prediction error for {vehicle_id}: {e}")
            return False
    
    def update_ground_truth(self, vehicle_ids: List[str]):
        """Update ground truth for all active prediction records"""
        for record_key, record in list(self.prediction_records.items()):
            vehicle_id = record.vehicle_id
            
            # If vehicle still exists, collect its position
            if vehicle_id in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vehicle_id)
                    record.add_ground_truth_frame(pos)
                except:
                    pass
    
    def collect_metrics(self):
        """Try to collect metrics from prediction records"""
        records_to_remove = []
        
        for record_key, record in list(self.prediction_records.items()):
            # Check if we can evaluate
            if record.can_evaluate(self.config.MIN_GT_FRAMES):
                result = record.get_metrics(self.config.PRED_LEN)
                
                if result is not None:
                    pred_traj, true_traj = result
                    
                    # Add to metrics
                    self.metrics.add_metric(
                        vehicle_id=record.vehicle_id,
                        prediction_step=record.prediction_step,
                        pred_traj=pred_traj,
                        true_traj=true_traj,
                        inference_ms=record.inference_ms,
                        e2e_ms=record.e2e_ms,
                        timestamp=record.timestamp
                    )
                    
                    self.total_metrics_collected += 1
                    records_to_remove.append(record_key)
                    continue
            
            # Check for timeout
            if record.should_timeout(self.config.MAX_GT_WAIT_STEPS):
                self.failed_collections += 1
                records_to_remove.append(record_key)
        
        # Remove completed/failed records
        for key in records_to_remove:
            del self.prediction_records[key]
    
    def update_visualizations(self, vehicle_ids: List[str]):
        """Update prediction visualizations"""
        # Clean up vehicles that left
        for vid in list(self.active_predictions.keys()):
            if vid not in vehicle_ids:
                self.clear_vehicle_visualization(vid)
                del self.active_predictions[vid]
        
        # Draw predictions
        if self.config.GUI and self.config.DRAW_PREDICTIONS:
            self.draw_predictions()
    
    def draw_predictions(self):
        """Draw prediction lines for active vehicles"""
        for vid, data in self.active_predictions.items():
            if vid not in traci.vehicle.getIDList():
                continue

            try:
                current_pos = traci.vehicle.getPosition(vid)
                angle = traci.vehicle.getAngle(vid)
                pred = data['prediction']
                display_len = min(self.config.PRED_DISPLAY_LEN, len(pred))

                traci.vehicle.setColor(vid, self.config.VEHICLE_COLOR)

                angle_rad = np.radians(90 - angle)
                perp_x = -np.sin(angle_rad) * self.config.LATERAL_OFFSET
                perp_y = np.cos(angle_rad) * self.config.LATERAL_OFFSET

                pred_rel = pred[:display_len] - pred[0]
                dx, dy = pred_rel[:, 0], pred_rel[:, 1]

                x_local = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                y_local = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

                pred_local = np.column_stack((x_local, y_local))

                self._draw_dashed_line(
                    vid, "pred",
                    current_pos[0] + perp_x,
                    current_pos[1] + perp_y,
                    pred_local,
                    self.config.PRED_LINE_COLOR
                )
            except:
                continue
    
    def _draw_dashed_line(self, vehicle_id, prefix, start_x, start_y, trajectory, color):
        """Draw a dashed line"""
        dash_len = self.config.DASH_LENGTH
        gap_len = self.config.DASH_GAP
        width = self.config.LINE_WIDTH

        seg_id = 0
        acc_dist = 0.0
        dash_active = True
        seg_points = []
        last_x, last_y = start_x, start_y

        for i in range(len(trajectory)):
            x = start_x + trajectory[i, 0]
            y = start_y + trajectory[i, 1]

            if dash_active:
                seg_points.append((x, y))

            dx, dy = x - last_x, y - last_y
            acc_dist += np.hypot(dx, dy)

            if dash_active and acc_dist >= dash_len:
                if len(seg_points) >= 2:
                    poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
                    self._safe_remove_object(poly_id)
                    try:
                        traci.polygon.add(poly_id, shape=seg_points, color=color, 
                                        fill=False, lineWidth=width, layer=102)
                        self.drawn_objects.add(poly_id)
                    except:
                        pass
                    seg_id += 1
                dash_active = False
                acc_dist = 0.0
                seg_points = []
            elif not dash_active and acc_dist >= gap_len:
                dash_active = True
                acc_dist = 0.0
                seg_points = [(x, y)]

            last_x, last_y = x, y
    
    def _safe_remove_object(self, obj_id: str):
        """Safely remove a drawn object"""
        if obj_id in self.drawn_objects:
            try:
                traci.polygon.remove(obj_id)
            except:
                pass
            self.drawn_objects.discard(obj_id)
    
    def clear_vehicle_visualization(self, vehicle_id: str):
        """Clear all visualizations for a vehicle"""
        for prefix in ["pred_", "true_"]:
            for i in range(200):
                self._safe_remove_object(f"{prefix}{vehicle_id}_{i}")
    
    def clear_all_visualizations(self):
        """Clear all visualizations"""
        for obj_id in list(self.drawn_objects):
            self._safe_remove_object(obj_id)


def run_dt_simulation(config: DTConfig):
    print("\n" + "="*70)
    print("DIGITAL TWIN SUMO SIMULATION - WORKING METRICS")
    print("="*70)
    print(f"Observation: {config.OBS_LEN} frames ({config.OBS_LEN * 0.25}s)")
    print(f"Prediction: {config.PRED_LEN} frames ({config.PRED_LEN * 0.25}s)")
    print(f"Min GT frames: {config.MIN_GT_FRAMES} ({config.MIN_GT_FRAMES * 0.25}s)")
    print("="*70 + "\n")
    
    # Load model
    model = load_model(config.MODEL_PATH, config.MODEL_TYPE, config.PRED_LEN)
    
    # Initialize components
    tracker = TrajectoryTracker(obs_len=config.OBS_LEN)
    metrics = SimpleMetricsCollector(config)
    predictor = DigitalTwinPredictor(model, tracker, metrics, config)
    
    # Import SUMO setup
    from main import (trajectory_tracking, aggregate_vehicles, gene_config, 
                     has_vehicle_entered, AVAILABLE_CAR_TYPES, AVAILABLE_TRUCK_TYPES,
                     CHECK_ALL, LAN_CHANGE_MODE)
    
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    
    cfg_file = gene_config()
    start_sumo(cfg_file + "/freeway.sumo.cfg", False, gui=config.GUI)
    
    times = 0
    random.seed(7)
    
    print("Starting simulation...\n")
    
    try:
        while running(True, times, config.TOTAL_TIME + 1):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Add vehicles
            if times > 0 and times % 4 == 0:
                current_step = int(times / 4)
                
                if has_vehicle_entered(current_step, vehicles_to_enter):
                    for data in vehicles_to_enter[current_step]:
                        vehicle_class = data["class"].lower()
                        
                        if "truck" in vehicle_class or "bus" in vehicle_class:
                            type_id = random.choice(AVAILABLE_TRUCK_TYPES)
                            depart_speed = random.uniform(24, 25)
                        else:
                            type_id = random.choice(AVAILABLE_CAR_TYPES)
                            depart_speed = random.uniform(31, 33)
                        
                        lane_id = max(0, min(2, int(data.get("laneId", 1)) - 1))
                        depart_pos = random.uniform(10, 30)
                        direction = data.get("drivingDirection", 1)
                        
                        route_id = "route_direction1" if direction == 1 else "route_direction2"
                        vehicle_id = f"d{direction}_{data['id']}"
                        
                        try:
                            traci.vehicle.add(
                                vehID=vehicle_id, routeID=route_id,
                                typeID=type_id, departSpeed=depart_speed,
                                departPos=depart_pos, departLane=lane_id,
                            )
                            traci.vehicle.setSpeedMode(vehicle_id, CHECK_ALL)
                            traci.vehicle.setLaneChangeMode(vehicle_id, LAN_CHANGE_MODE)
                        except:
                            pass
            
            vehicle_ids = traci.vehicle.getIDList()
            
            # Update trajectory history for ALL vehicles
            for vid in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    acc = traci.vehicle.getAcceleration(vid)
                    lane = traci.vehicle.getLaneIndex(vid)
                    tracker.update(vid, pos, speed, acc, lane)
                except:
                    continue
            
            # Digital twin operations
            if times > config.START_STEP:
                # Make predictions for eligible vehicles
                for vid in vehicle_ids:
                    if (tracker.has_enough_history(vid) and 
                        predictor.should_predict(vid, current_time)):
                        predictor.make_prediction(vid, current_time, times)
                
                # Update ground truth for active prediction records
                predictor.update_ground_truth(vehicle_ids)
                
                # Collect metrics (every step)
                predictor.collect_metrics()
                
                # Update visualizations
                predictor.update_visualizations(vehicle_ids)
            
            # Status updates
            if times % 500 == 0 and times > 0:
                num_vehicles = len(vehicle_ids)
                num_displaying = len(predictor.active_predictions)
                num_records = len(predictor.prediction_records)
                num_metrics = len(metrics.metrics)
                
                if num_metrics > 0:
                    recent = metrics.metrics[-10:]
                    avg_ade = np.mean([m.ade for m in recent])
                    avg_fde = np.mean([m.fde for m in recent])
                    avg_lat = np.mean([m.inference_latency_ms for m in recent])
                    
                    print(f"Step {times:5d} | Vehicles: {num_vehicles:3d} | "
                          f"Displaying: {num_displaying:3d} | Pending: {num_records:3d} | "
                          f"Metrics: {num_metrics:4d} | "
                          f"ADE: {avg_ade:.2f}m | FDE: {avg_fde:.2f}m | Lat: {avg_lat:.2f}ms")
                else:
                    print(f"Step {times:5d} | Vehicles: {num_vehicles:3d} | "
                          f"Displaying: {num_displaying:3d} | Pending: {num_records:3d} | "
                          f"Metrics: {num_metrics:4d} (collecting...)")
            
            if times >= config.TOTAL_TIME:
                print(f"\n✓ Simulation complete at step {times}")
                break
            
            times += 1
    
    except KeyboardInterrupt:
        print("\n⏸ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.clear_all_visualizations()
        try:
            traci.close()
        except:
            pass
        time.sleep(0.5)
    
    # Final summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    
    summary = metrics.get_summary()
    
    if summary:
        print(f"\n📊 Results ({len(metrics.metrics)} predictions):")
        print(f"  Total predictions made: {predictor.total_predictions_made}")
        print(f"  Metrics collected: {predictor.total_metrics_collected}")
        print(f"  Failed collections: {predictor.failed_collections}")
        print(f"  Collection rate: {predictor.total_metrics_collected/predictor.total_predictions_made*100:.1f}%")
        
        acc = summary['trajectory_accuracy']
        print(f"\n  ADE: {acc['ADE_mean']:.3f} ± {acc['ADE_std']:.3f} m")
        print(f"       (median: {acc['ADE_median']:.3f}, range: [{acc['ADE_min']:.3f}, {acc['ADE_max']:.3f}])")
        print(f"  FDE: {acc['FDE_mean']:.3f} ± {acc['FDE_std']:.3f} m")
        print(f"       (median: {acc['FDE_median']:.3f}, range: [{acc['FDE_min']:.3f}, {acc['FDE_max']:.3f}])")
        
        lat = summary['latency_metrics']
        print(f"\n  Inference: {lat['inference_mean_ms']:.2f} ± {lat['inference_std_ms']:.2f} ms")
        print(f"             (P95: {lat['inference_p95_ms']:.2f} ms, P99: {lat['inference_p99_ms']:.2f} ms)")
        print(f"  E2E: {lat['e2e_mean_ms']:.2f} ± {lat['e2e_std_ms']:.2f} ms")
        
        metrics.save_results(config.METRICS_OUTPUT)
    else:
        print("\n⚠️  No metrics collected!")
        print(f"  Predictions made: {predictor.total_predictions_made}")
        print(f"  Active records: {len(predictor.prediction_records)}")
    
    print("\n" + "="*70)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fixed Digital Twin SUMO Simulation")
    parser.add_argument("--mode", choices=["dt", "baseline"], default="dt")
    parser.add_argument("--model_path", type=str,
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--total_time", type=int, default=4000)
    parser.add_argument("--output", type=str, default="./dt_metrics_fixed.json")
    parser.add_argument("--min_frames", type=int, default=10, help="Minimum frames for metric collection")

    args = parser.parse_args()

    config.USE_DT_PREDICTION = (args.mode == "dt")
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.GUI = args.gui
    config.TOTAL_TIME = args.total_time
    config.METRICS_OUTPUT = args.output
    config.MIN_FRAMES_FOR_METRICS = args.min_frames

    run_dt_simulation(config)