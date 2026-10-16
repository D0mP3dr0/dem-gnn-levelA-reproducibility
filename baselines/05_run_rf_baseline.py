INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
INTERNAL_GRAPH_ROOT = __import__("os").environ.get("DEMGNN_GRAPH_ROOT", ".")
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import time
import torch
import sys

BASE_DIR = Path(f"{INTERNAL_DATA_ROOT}")
DATA_DIR = BASE_DIR / "CNN_U-NET" / "data" / "data_cnn_v2"
OUT_DIR = BASE_DIR / "kriging/results"
STATS_FILE = BASE_DIR / "v21/GNN_RUN_COLAB/config.py"
GRAPH_BASE = Path(f"{INTERNAL_GRAPH_ROOT}/GINE_GRAPH")

sys.path.append(str(BASE_DIR / "CNN_U-NET/ResU-Net_RUN_COLAB"))
from src.config import STATS as OFFICIAL_STATS, LEGACY_STATS

def load_and_sample_training_data(biome_path, biome, n_tiles=50, n_pts_per_tile=5000):
    input_files = sorted(list(biome_path.glob("*_input.npy")))
    train_files = input_files[:len(input_files)//2]
    train_files = train_files[:n_tiles]
    
    legacy = LEGACY_STATS.get(biome, LEGACY_STATS.get('amazonia'))
    official = OFFICIAL_STATS.get(biome, OFFICIAL_STATS['default'])
    o_std, o_mean = legacy['std'], legacy['mean']
    n_std, n_mean = official['std'], official['mean']

    X_train, y_train = [], []
    print(f"[RF] Loading training data for {biome.upper()}...")
    
    for f in train_files:
        try:
            inp = np.load(f)
            tgt = np.load(str(f).replace("_input.npy", "_target.npy"))
            inp[:, :, 0] = (inp[:, :, 0] * o_std + o_mean - n_mean) / n_std
            tgt[:, :, 0] = (tgt[:, :, 0] * o_std + o_mean - n_mean) / n_std
            mask = tgt[:, :, 1] > 0.5
            if not mask.any(): continue
            y_vals, X_vals = tgt[mask, 0], inp[mask, :17] 
            if len(y_vals) > n_pts_per_tile:
                idx = np.random.choice(len(y_vals), n_pts_per_tile, replace=False)
                X_train.append(X_vals[idx]); y_train.append(y_vals[idx])
            else:
                X_train.append(X_vals); y_train.append(y_vals)
        except Exception: continue
    return np.vstack(X_train), np.concatenate(y_train)

def get_golden_sector_tiles(biome):
    graph_path = GRAPH_BASE / f"{biome}_v18.3_part2.pt"
    if not graph_path.exists(): return list(range(120))
    data = torch.load(str(graph_path), map_location="cpu", weights_only=False)
    coords, l_pos = data['dem'].pos, data['lidar'].pos
    min_x, max_x = coords[:, 0].min().item(), coords[:, 0].max().item()
    min_y, max_y = coords[:, 1].min().item(), coords[:, 1].max().item()
    nx, ny = 4, 4
    dx, dy = (max_x - min_x) / nx, (max_y - min_y) / ny
    max_lidar, best_sector = -1, None
    for ix in range(nx):
        for iy in range(ny):
            bx0, bx1 = min_x + ix*dx, min_x + (ix+1)*dx
            by0, by1 = min_y + iy*dy, min_y + (iy+1)*dy
            count = ((l_pos[:,0] >= bx0) & (l_pos[:,0] < bx1) & (l_pos[:,1] >= by0) & (l_pos[:,1] < by1)).sum().item()
            if count > max_lidar:
                max_lidar, best_sector = count, (bx0, bx1, by0, by1)
    gx, gy = 12, 10
    gdx, gdy = (max_x - min_x) / gx, (max_y - min_y) / gy
    golden_indices = []
    bx0, bx1, by0, by1 = best_sector
    for idx in range(120):
        row, col = idx // 12, idx % 12
        tx, ty = min_x + col * gdx + gdx/2, min_y + row * gdy + gdy/2
        if bx0 <= tx <= bx1 and by0 <= ty <= by1: golden_indices.append(idx)
    return golden_indices

def run_rf_evaluation(biome, mode='full'):
    t_start = time.time()
    official = OFFICIAL_STATS.get(biome, OFFICIAL_STATS['default'])
    elev_mean, elev_std = official['mean'], official['std']
    legacy = LEGACY_STATS.get(biome, LEGACY_STATS.get('amazonia'))
    o_std, o_mean = legacy['std'], legacy['mean']
    
    out_path = OUT_DIR / f"rf_{biome}_{mode}_metrics.csv"
    if out_path.exists():
        print(f"[SKIP] RF {biome.upper()} ({mode.upper()}) already exists."); return

    biome_path = DATA_DIR / biome
    X_train, y_train = load_and_sample_training_data(biome_path, biome)
    print(f"[RF] Ajustando RandomForest...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    input_files = sorted(list(biome_path.glob("*_input.npy")))
    val_files_all = input_files[len(input_files)//2:] if len(input_files) > 120 else input_files
    if mode == 'golden':
        golden_indices = get_golden_sector_tiles(biome)
        val_files = [val_files_all[i] for i in golden_indices if i < len(val_files_all)]
    else:
        val_files = val_files_all[:120]

    results = []
    print(f"[RF] Avaliando {len(val_files)} tiles ({mode.upper()})...")
    for i, f in enumerate(val_files):
        inp = np.load(f)
        tgt = np.load(str(f).replace("_input.npy", "_target.npy"))
        inp[:, :, 0] = (inp[:, :, 0] * o_std + o_mean - elev_mean) / elev_std
        tgt[:, :, 0] = (tgt[:, :, 0] * o_std + o_mean - elev_mean) / elev_std
        mask_gt = tgt[:, :, 1] > 0.5
        if not mask_gt.any(): continue
        mask_anchor = inp[:, :, 14] > 0.5
        mask_gap, mask_anchor_eval = mask_gt & (~mask_anchor), mask_gt & mask_anchor
        abs_err = np.abs(rf.predict(inp[:,:,:17].reshape(-1, 17)).reshape(inp.shape[0], inp.shape[1]) * elev_std - tgt[:,:,0] * elev_std)
        mae_global = np.mean(abs_err[mask_gt])
        mae_anchor = np.mean(abs_err[mask_anchor_eval]) if mask_anchor_eval.any() else mae_global
        mae_gap = np.mean(abs_err[mask_gap]) if mask_gap.any() else mae_global
        results.append({"biome": biome, "tile": f.name, "mae_m": mae_global, "mae_anchor": mae_anchor, "mae_gap": mae_gap, "gap_ratio": mae_gap / mae_anchor if mae_anchor > 0 else 1.0})
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[{biome.upper()}] Finished. Saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--biome", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['full', 'golden'], default='full')
    args = parser.parse_args()
    run_rf_evaluation(args.biome, args.mode)
