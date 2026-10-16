INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
INTERNAL_GRAPH_ROOT = __import__("os").environ.get("DEMGNN_GRAPH_ROOT", ".")
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import gc
import random

CNN_ROOT = Path(rf"{INTERNAL_DATA_ROOT}\CNN_U-NET\ResU-Net_RUN_COLAB")
sys.path.insert(0, str(CNN_ROOT))

GNN_DATA_DIR = Path(rf"{INTERNAL_GRAPH_ROOT}\GINE_GRAPH")

from src.models.unet_model import CNNTrainer
from src.config import STATS as CNN_STATS

GRID_H = 1800
GRID_W = 7200
PATCH_SIZE = 512

GNN_NORM_STATS = {
    'amazonia':      {'elev_mean': 78.25,   'elev_std': 35.28},
    'mata_atlantica':{'elev_mean': 363.58,  'elev_std': 76.22},
    'cerrado':       {'elev_mean': 759.63,  'elev_std': 240.27},
    'pantanal':      {'elev_mean': 106.71,  'elev_std': 28.74},
    'default':       {'elev_mean': 500.0,   'elev_std': 300.0}
}


def get_checkpoint_for_biome(biome, use_lidar_input=False):
    suffix = "lidar" if use_lidar_input else "sr"
    ckpt_dir = CNN_ROOT / "checkpoints" / f"{biome}_{suffix}"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint for biome {biome} in {ckpt_dir}")
    return ckpts[0]

def run_evaluation(biome, max_pt_files=30, patches_per_file=10, use_lidar_input=False):
    suffix = "lidar" if use_lidar_input else "sr"
    print(f"\n{'='*60}")
    print(f"  SCIENTIFICALLY FAIR CNN EVALUATION (UNBIASED SAMPLING)")
    print(f"  BIOME: {biome.upper()} | MODE: {suffix.upper()}")
    print(f"{'='*60}")
    
    ckpt_path = get_checkpoint_for_biome(biome, use_lidar_input)
    print(f"[1/4] Loading model: {ckpt_path.name}")
    model = CNNTrainer.load_from_checkpoint(str(ckpt_path), map_location="cpu", use_lidar_input=use_lidar_input)
    model.eval()
    
    stats = CNN_STATS.get(biome, {'mean': 0.0, 'std': 1.0})
    std_val = stats['std']
    mean_val = stats['mean']
    
    graph_files = list(GNN_DATA_DIR.glob(f"{biome}_v18.3_part*.pt"))
    if not graph_files:
        print(f"Error: no .pt file found for {biome}")
        return
    
    graph_files.sort()
    graph_files = graph_files[:max_pt_files]
    
    results = []
    processed_patches = 0
    t0_total = time.perf_counter()
    
    print(f"[3/4] Starting blind sampling (10 patches per file)...")
    
    for graph_file in graph_files:
        print(f"\n  Lendo: {graph_file.name}")
        try:
            data = torch.load(graph_file, map_location="cpu", weights_only=False)
            
            if hasattr(data, 'node_types'):
                x = data['dem'].x
                y_true = data['dem'].y
            else:
                x = data.x
                y_true = data.y
            
            curr_ch = 19 if use_lidar_input else 17
            x_grid = x.view(GRID_H, GRID_W, 17)
            
            
            y_grid = y_true[:, 0].view(GRID_H, GRID_W)
            mask_grid = x[:, 14].view(GRID_H, GRID_W)
            
            gnn_norm = GNN_NORM_STATS.get(biome, GNN_NORM_STATS['default'])
            cnn_norm = CNN_STATS.get(biome)
            
            g_mean, g_std = gnn_norm['elev_mean'], gnn_norm['elev_std']
            c_mean, c_std = cnn_norm['mean'], cnn_norm['std']

            for p_idx in range(patches_per_file):
                r_start = random.randint(0, GRID_H - PATCH_SIZE)
                c_start = random.randint(0, GRID_W - PATCH_SIZE)
                
                x_patch_gnn = x_grid[r_start:r_start+PATCH_SIZE, c_start:c_start+PATCH_SIZE, :]
                y_patch_gnn = y_grid[r_start:r_start+PATCH_SIZE, c_start:c_start+PATCH_SIZE]
                m_patch = mask_grid[r_start:r_start+PATCH_SIZE, c_start:c_start+PATCH_SIZE]
                
                in_dem_gnn = x_patch_gnn[:, :, 0]
                in_dem_m = in_dem_gnn * g_std + g_mean
                target_m = y_patch_gnn * g_std + g_mean
                
                in_dem_cnn = (in_dem_m - c_mean) / c_std
                
                x_input_cnn = x_patch_gnn.clone()
                x_input_cnn[:, :, 0] = in_dem_cnn
                
                if use_lidar_input:
                    lidar_mask = x_patch_gnn[:, :, 14:15]
                    lidar_sparse_cnn = ((target_m.unsqueeze(-1) - c_mean) / c_std) * lidar_mask
                    x_input_final = torch.cat([x_input_cnn, lidar_sparse_cnn, lidar_mask], dim=-1)
                else:
                    x_input_final = x_input_cnn
                
                x_input_tensor = x_input_final.permute(2, 0, 1).unsqueeze(0).to(model.device)
                
                with torch.no_grad():
                    pred_res_cnn = model(x_input_tensor).squeeze().cpu()
                
                pred_final_m = in_dem_m + (pred_res_cnn * c_std) 
                
                abs_err = torch.abs(pred_final_m - target_m)
                
                has_lidar = m_patch > 0.5
                no_lidar = ~has_lidar
                
                p_mae = abs_err[m_patch > 0].mean().item() if (m_patch > 0).any() else 0.0
                p_anchor = abs_err[has_lidar].mean().item() if has_lidar.any() else p_mae
                p_gap = abs_err[no_lidar].mean().item() if no_lidar.any() else p_mae
                
                results.append({
                    'file': graph_file.name,
                    'mode': suffix,
                    'mae_m': p_mae,
                    'mae_anchor': p_anchor,
                    'mae_gap': p_gap
                })
                
                processed_patches += 1
            
            del data, x, y_true, x_grid, y_grid, mask_grid, input_dem_grid
            gc.collect()
            
        except Exception as e:
            print(f"    Error while processing {graph_file.name}: {e}")
            continue

    df = pd.DataFrame(results)
    out_path = Path(f"cnn_{biome}_{suffix}_metrics.csv")
    df.to_csv(out_path, index=False)
    
    elapsed = (time.perf_counter() - t0_total) / 60
    
    print(f"\n[4/4] Evaluation finished ({elapsed:.1f} min)")
    print(f"  - Modo: {suffix.upper()}")
    print(f"  - Mean gap MAE: {df['mae_gap'].mean():.4f} m")
    print(f"Results saved to: {out_path.absolute()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("biome", type=str, help="Bioma")
    parser.add_argument("--max_files", type=int, default=30)
    parser.add_argument("--patches", type=int, default=10)
    parser.add_argument("--use_lidar_input", action="store_true")
    args = parser.parse_args()
    
    run_evaluation(args.biome, max_pt_files=args.max_files, patches_per_file=args.patches, use_lidar_input=args.use_lidar_input)
