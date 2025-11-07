# highd_dataloader_fixed.py
"""
Fixed and multi-scene-ready HighD dataloader.
- make_dataloader_highd(path_or_df, ...) accepts path (CSV) or DataFrame
- make_dataloader_highd_multiscene(file_list, ...) returns DataLoader over concatenated scenes
- HighDDatasetFixed, collate_fn_fixed unchanged in API (compatible)
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# defaults
DOWNSAMPLE = 5
OBS_SECONDS = 2.0
PRED_SECONDS = 5.0
OBS_LEN = int(OBS_SECONDS * 5)
PRED_LEN = int(PRED_SECONDS * 5)
K_NEIGH = 8
MAX_SPEED = 50.0

# ---------------- utilities ----------------

def load_highd_scenes(folder_path, train_ratio=0.8):
    """
    List scene CSV files in folder and split into train/test file lists.
    Expects filenames matching '*_tracks.csv'.
    """
    csv_files = sorted([os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if f.endswith("_tracks.csv")])
    if len(csv_files) == 0:
        raise ValueError(f"No '*_tracks.csv' files found in {folder_path}")
    n_train = int(len(csv_files) * train_ratio)
    train_files = csv_files[:n_train]
    test_files = csv_files[n_train:]
    print(f"Detected {len(csv_files)} scenes. Using {n_train} for training, {len(test_files)} for validation.")
    return train_files, test_files

def make_dataloader_highd_multiscene(file_list, batch_size=32, obs_len=OBS_LEN, pred_len=PRED_LEN, shuffle=True, **kwargs):
    """
    Combine multiple scene datasets into one DataLoader.
    Each entry in file_list is a CSV path. Uses make_dataloader_highd for each.
    Returns a single DataLoader over a ConcatDataset.
    """
    datasets = []
    for path in file_list:
        loader = make_dataloader_highd(path, batch_size=1, obs_len=obs_len, pred_len=pred_len, shuffle=False, **kwargs)
        datasets.append(loader.dataset)
    if len(datasets) == 0:
        raise ValueError("No datasets found in file_list")
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_fixed, drop_last=True)

# ---------------- feature helpers ----------------

def compute_8dir_features(target_pos, target_vel, target_acc, target_heading,
                          neigh_pos, neigh_vel, neigh_acc, neigh_heading):
    """Compute 8-directional spatial relationship features for a single time step pair."""
    dx = neigh_pos[0] - target_pos[0]
    dy = neigh_pos[1] - target_pos[1]
    dist = math.hypot(dx, dy) + 1e-6

    angle = math.atan2(dy, dx)
    angle_deg = (math.degrees(angle) + 360.0) % 360.0

    D_dirs = np.zeros(8, dtype=np.float32)
    V_dirs = np.zeros(8, dtype=np.float32)

    bin_idx = int((angle_deg + 22.5) // 45) % 8
    D_dirs[bin_idx] = dist

    dvx = neigh_vel[0] - target_vel[0]
    dvy = neigh_vel[1] - target_vel[1]
    V_dirs[bin_idx] = math.hypot(dvx, dvy)

    dax = neigh_acc[0] - target_acc[0]
    day = neigh_acc[1] - target_acc[1]
    delta_a = math.hypot(dax, day)

    delta_phi = neigh_heading - target_heading
    delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi

    return np.concatenate([D_dirs, V_dirs, [delta_a, delta_phi]]).astype(np.float32)

# ---------------- dataset ----------------

class HighDDatasetFixed(Dataset):
    """
    Dataset that returns agent-centric sequences:
    - 'target_feats': (T_obs, 7) [x,y,vx,vy,ax,ay,heading] in agent frame (last obs at origin)
    - 'neighbors_dyn': (K, T_obs, 7)
    - 'neighbors_spatial': (K, T_obs, 18)
    - 'lane_feats': (T_obs, 1)
    - 'gt': (T_pred, 2) future positions relative to last obs (origin)
    - 'meta': dict
    """
    def __init__(self, tracks_df, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 downsample=DOWNSAMPLE, k_neighbors=K_NEIGH, max_speed=MAX_SPEED):
        super().__init__()
        if isinstance(tracks_df, str):
            tracks_df = pd.read_csv(tracks_df)
        self.tracks = tracks_df
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.down = downsample
        self.k = k_neighbors
        self.max_speed = max_speed

        # group by vehicle id (support both 'id' and 'Vehicle_ID' column names)
        id_col = 'id' if 'id' in tracks_df.columns else ('Vehicle_ID' if 'Vehicle_ID' in tracks_df.columns else None)
        if id_col is None:
            raise ValueError("No vehicle id column found ('id' or 'Vehicle_ID')")
        self.groups = {}
        for vid, g in tracks_df.groupby(id_col):
            df = g.sort_values(by=g.columns[0]).copy()  # sort by first column (frame)
            # ensure heading column exists
            if 'heading' not in df.columns:
                if 'yVelocity' in df.columns and 'xVelocity' in df.columns:
                    df['heading'] = np.arctan2(df['yVelocity'].values, df['xVelocity'].values)
                elif 'vy' in df.columns and 'vx' in df.columns:
                    df['heading'] = np.arctan2(df['vy'].values, df['vx'].values)
                else:
                    df['heading'] = 0.0
            # normalize column names to expected indices
            # attempt to find required columns robustly
            cols = list(df.columns)
            # map common names to expected order: frame,x,y,xVelocity,yVelocity,xAcceleration,yAcceleration,heading,laneId
            col_map = {
                'frame': None, 'x': None, 'y': None, 'xVelocity': None, 'yVelocity': None,
                'xAcceleration': None, 'yAcceleration': None, 'heading': None, 'laneId': None,
                'vx': None, 'vy': None, 'ax': None, 'ay': None, 'Vehicle_ID': None
            }
            for c in cols:
                if c in col_map:
                    col_map[c] = c
            # fallback heuristics: try common alt names
            if col_map['xVelocity'] is None and 'vx' in cols:
                col_map['xVelocity'] = 'vx'
            if col_map['yVelocity'] is None and 'vy' in cols:
                col_map['yVelocity'] = 'vy'
            if col_map['xAcceleration'] is None and 'ax' in cols:
                col_map['xAcceleration'] = 'ax'
            if col_map['yAcceleration'] is None and 'ay' in cols:
                col_map['yAcceleration'] = 'ay'
            if col_map['laneId'] is None and 'laneId' in cols:
                col_map['laneId'] = 'laneId'
            # build array with safe indexing (fill missing with zeros)
            arr_cols = []
            for key in ['frame','x','y','xVelocity','yVelocity','xAcceleration','yAcceleration','heading','laneId']:
                if col_map.get(key) is not None:
                    arr_cols.append(df[col_map[key]].values)
                else:
                    arr_cols.append(np.zeros(len(df)))
            arr = np.stack(arr_cols, axis=1)
            self.groups[vid] = arr

        # build valid sliding-window samples
        self.samples = []
        for vid, arr in self.groups.items():
            n = len(arr)
            step = self.down
            total_needed = (self.obs_len + self.pred_len) * step
            if n < total_needed:
                continue
            # sliding with stride = step for coverage
            for start in range(0, n - total_needed + 1, step):
                self.samples.append((vid, start))
        if len(self.samples) == 0:
            raise ValueError("No valid samples in dataset. Check obs_len/pred_len/downsample vs track lengths.")

    def __len__(self):
        return len(self.samples)

    def _get_window(self, vid, start):
        arr = self.groups[vid]
        idxs = np.arange(start, start + (self.obs_len + self.pred_len) * self.down, self.down).astype(int)
        return arr[idxs, :]

    def __getitem__(self, idx):
        vid, start = self.samples[idx]
        window = self._get_window(vid, start)
        frames = window[:, 0].astype(int)
        x, y = window[:, 1], window[:, 2]
        vx, vy = window[:, 3], window[:, 4]
        ax, ay = window[:, 5], window[:, 6]
        heading = window[:, 7]
        lane_id = window[:, 8]

        # clip extreme speeds
        speeds = np.sqrt(vx**2 + vy**2)
        if np.any(speeds > self.max_speed):
            factor = np.minimum(1.0, self.max_speed / (speeds + 1e-6))
            vx *= factor
            vy *= factor

        obs_x, obs_y = x[:self.obs_len], y[:self.obs_len]
        fut_x, fut_y = x[self.obs_len:], y[self.obs_len:]

        origin = np.array([obs_x[-1], obs_y[-1]], dtype=np.float32)
        yaw = float(heading[self.obs_len - 1])

        def transform(xy_arr, vel_arr=None, acc_arr=None, head_arr=None):
            c, s = math.cos(-yaw), math.sin(-yaw)
            dx = xy_arr[:, 0] - origin[0]
            dy = xy_arr[:, 1] - origin[1]
            xr = c*dx - s*dy
            yr = s*dx + c*dy
            results = [np.stack([xr, yr], axis=-1).astype(np.float32)]
            if vel_arr is not None:
                vx_r = c*vel_arr[:, 0] - s*vel_arr[:, 1]
                vy_r = s*vel_arr[:, 0] + c*vel_arr[:, 1]
                results.append(np.stack([vx_r, vy_r], axis=-1).astype(np.float32))
            if acc_arr is not None:
                ax_r = c*acc_arr[:, 0] - s*acc_arr[:, 1]
                ay_r = s*acc_arr[:, 0] + c*acc_arr[:, 1]
                results.append(np.stack([ax_r, ay_r], axis=-1).astype(np.float32))
            if head_arr is not None:
                head_rel = ((head_arr - yaw + np.pi) % (2*np.pi) - np.pi).astype(np.float32)
                results.append(head_rel)
            return results

        obs_xy = np.stack([obs_x, obs_y], axis=-1)
        fut_xy = np.stack([fut_x, fut_y], axis=-1)
        obs_vel = np.stack([vx[:self.obs_len], vy[:self.obs_len]], axis=-1)
        obs_acc = np.stack([ax[:self.obs_len], ay[:self.obs_len]], axis=-1)

        obs_xy_rel, obs_vel_rel, obs_acc_rel, obs_head_rel = transform(obs_xy, obs_vel, obs_acc, heading[:self.obs_len])
        fut_xy_rel = transform(fut_xy)[0]

        # verify last obs is origin in agent frame (small numerical tolerance)
        assert np.allclose(obs_xy_rel[-1], [0.0, 0.0], atol=1e-4), "Last obs should be at origin!"

        # target features
        target_feats = np.concatenate([
            obs_xy_rel,
            obs_vel_rel,
            obs_acc_rel,
            obs_head_rel.reshape(-1, 1)
        ], axis=-1).astype(np.float32)

        # neighbors
        last_frame = frames[self.obs_len - 1]
        neighbors_dyn = np.zeros((self.k, self.obs_len, 7), dtype=np.float32)
        neighbors_spatial = np.zeros((self.k, self.obs_len, 18), dtype=np.float32)

        candidates = []
        for other_vid, other_arr in self.groups.items():
            if other_vid == vid:
                continue
            matching = np.where(other_arr[:, 0].astype(int) == last_frame)[0]
            if matching.size == 0:
                continue
            idx = int(matching[0])
            if idx - (self.obs_len - 1) * self.down < 0:
                continue
            idxs = np.arange(idx - (self.obs_len - 1) * self.down, idx + 1, self.down).astype(int)
            neigh_win = other_arr[idxs, :]

            n_xy = neigh_win[:, 1:3]
            n_vel = neigh_win[:, 3:5]
            n_acc = neigh_win[:, 5:7]
            n_head = neigh_win[:, 7]

            n_xy_rel, n_vel_rel, n_acc_rel, n_head_rel = transform(n_xy, n_vel, n_acc, n_head)

            dyn_feats = np.concatenate([n_xy_rel, n_vel_rel, n_acc_rel, n_head_rel.reshape(-1, 1)], axis=-1)

            spatial_feats = []
            for t in range(self.obs_len):
                feat_8d = compute_8dir_features(
                    obs_xy_rel[t], obs_vel_rel[t], obs_acc_rel[t], obs_head_rel[t],
                    n_xy_rel[t], n_vel_rel[t], n_acc_rel[t], n_head_rel[t]
                )
                spatial_feats.append(feat_8d)
            spatial_feats = np.array(spatial_feats, dtype=np.float32)

            dist = float(np.linalg.norm(n_xy_rel[-1], ord=2))
            candidates.append((dist, dyn_feats, spatial_feats))

        candidates.sort(key=lambda x: x[0])
        for i, (d, dyn, spatial) in enumerate(candidates[:self.k]):
            neighbors_dyn[i] = dyn
            neighbors_spatial[i] = spatial

        lane_feats = (lane_id[:self.obs_len] / 10.0).reshape(-1, 1).astype(np.float32)

        return {
            'target_feats': target_feats,  # (T_obs, 7)
            'neighbors_dyn': neighbors_dyn,
            'neighbors_spatial': neighbors_spatial,
            'lane_feats': lane_feats,
            'gt': fut_xy_rel.astype(np.float32),  # (T_pred, 2)
            'meta': {
                'vid': vid,
                'start': start,
                'origin': origin,
                'yaw': yaw,
                'obs_world': obs_xy,
                'fut_world': fut_xy
            }
        }

# ---------------- collate / dataloader ----------------

def collate_fn_fixed(batch):
    B = len(batch)
    obs_len = batch[0]['target_feats'].shape[0]
    pred_len = batch[0]['gt'].shape[0]
    K = batch[0]['neighbors_dyn'].shape[0]

    target = np.zeros((B, obs_len, 7), dtype=np.float32)
    neigh_dyn = np.zeros((B, K, obs_len, 7), dtype=np.float32)
    neigh_spatial = np.zeros((B, K, obs_len, 18), dtype=np.float32)
    lane = np.zeros((B, obs_len, 1), dtype=np.float32)
    gt = np.zeros((B, pred_len, 2), dtype=np.float32)
    metas = []

    for i, s in enumerate(batch):
        target[i] = s['target_feats']
        neigh_dyn[i] = s['neighbors_dyn']
        neigh_spatial[i] = s['neighbors_spatial']
        lane[i] = s['lane_feats']
        gt[i] = s['gt']
        metas.append(s['meta'])

    return {
        'target': torch.from_numpy(target),
        'neigh_dyn': torch.from_numpy(neigh_dyn),
        'neigh_spatial': torch.from_numpy(neigh_spatial),
        'lane': torch.from_numpy(lane),
        'gt': torch.from_numpy(gt),
        'meta': metas
    }

def make_dataloader_fixed(tracks_df, batch_size=32, shuffle=True, **kwargs):
    """
    tracks_df may be a pandas DataFrame or a path to a CSV file.
    """
    ds = HighDDatasetFixed(tracks_df, **kwargs) if not isinstance(tracks_df, str) else HighDDatasetFixed(pd.read_csv(tracks_df), **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_fixed)

# backward-compatible alias
def make_dataloader_highd(path_or_df, batch_size=32, obs_len=OBS_LEN, pred_len=PRED_LEN, shuffle=True, **kwargs):
    """
    Backward-compatible loader: accepts CSV path or DataFrame.
    """
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df
    return make_dataloader_fixed(df, batch_size=batch_size, shuffle=shuffle, obs_len=obs_len, pred_len=pred_len, **kwargs)
