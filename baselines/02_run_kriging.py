INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
INTERNAL_GRAPH_ROOT = __import__("os").environ.get("DEMGNN_GRAPH_ROOT", ".")
import os
import torch
import numpy as np
from pykrige.ok import OrdinaryKriging
import pandas as pd
import time
import sys
import gc
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = f"{INTERNAL_GRAPH_ROOT}/GINE_GRAPH"
OUT_DIR = f"{INTERNAL_DATA_ROOT}/kriging/results"
os.makedirs(OUT_DIR, exist_ok=True)

PART = "part4"
BIOMES = ["amazonia", "cerrado", "mata_atlantica", "pantanal"]

GNN_STATS = {
    'amazonia':      {'elev_mean': 78.25,   'elev_std': 35.28},
    'mata_atlantica':{'elev_mean': 363.58,  'elev_std': 76.22},
    'cerrado':       {'elev_mean': 350.64,  'elev_std': 76.40},
    'pantanal':      {'elev_mean': 106.71,  'elev_std': 28.74},
    'default':       {'elev_mean': 500.0,   'elev_std': 300.0}
}

LEGACY_STATS = {
    'amazonia':      {'mean': 199.64,  'std': 122.67},
    'cerrado':       {'mean': 345.12,  'std': 180.45},
    'mata_atlantica':{'mean': 512.33,  'std': 210.88},
    'pantanal':      {'mean': 98.45,   'std': 45.22}
}

def get_device():
    if torch.cuda.is_available():
        try:
            test_t = torch.zeros(1, device="cuda")
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            return torch.device("cuda")
        except Exception as e:
            print(f"\n[KERNEL] WARNING: GPU detected but INCOMPATIBLE with this PyTorch ({e})")
            print('[KERNEL] Falling back automatically to CPU to ensure execution...\n')
            return torch.device("cpu")
    return torch.device("cpu")

def torch_kriging_solve(l_pos, l_z, d_pos, v_params, batch_size=2048, n_closest=64):
    device = l_pos.device
    psill, v_range, nugget = v_params
    
    num_targets = d_pos.shape[0]
    predictions = torch.zeros(num_targets, device=device)
    
    for i in range(0, num_targets, batch_size):
        batch_d_pos = d_pos[i:i+batch_size]
        curr_batch_size = batch_d_pos.shape[0]
        
        
        dist_to_neighbors = torch.cdist(batch_d_pos, l_pos)
        top_dist, top_idx = torch.topk(dist_to_neighbors, k=n_closest, largest=False)
        
        neighbor_pos = l_pos[top_idx] 
        neighbor_z = l_z[top_idx]
        
        dist_matrix = torch.cdist(neighbor_pos, neighbor_pos)
        
        h_ratio = (dist_matrix / v_range).clamp(max=1.0)
        cov_matrix = psill * (1.0 - (1.5 * h_ratio - 0.5 * h_ratio**3))
        
        h_target = (top_dist / v_range).clamp(max=1.0)
        target_cov = psill * (1.0 - (1.5 * h_target - 0.5 * h_target**3))
        
        A = torch.zeros((curr_batch_size, n_closest + 1, n_closest + 1), device=device)
        A[:, :n_closest, :n_closest] = cov_matrix
        
        eps = psill * 1e-2
        indices = torch.arange(n_closest, device=device)
        A[:, indices, indices] += eps
        
        A[:, n_closest, :n_closest] = 1.0
        A[:, :n_closest, n_closest] = 1.0
        
        b = torch.ones((curr_batch_size, n_closest + 1, 1), device=device)
        b[:, :n_closest, 0] = target_cov
        
        try:
            w = torch.linalg.solve(A, b)
            weights = w[:, :n_closest, 0]
            
            
            predictions[i:i+curr_batch_size] = (weights * neighbor_z).sum(dim=1)
            
            if i % 10000 == 0 and i > 0:
                print(f"    [BATCH] {i}/{num_targets} pontos processados...")
        except Exception:
            inv_dist = 1.0 / (top_dist + 1e-6)
            idw_weights = inv_dist / inv_dist.sum(dim=1, keepdim=True)
            predictions[i:i+curr_batch_size] = (idw_weights * neighbor_z).sum(dim=1)

    return predictions

def run_kriging_integrated(biome_input=None, mode='full', start_tile=0, end_tile=120):
    device = get_device()
    print(f"[KERNEL] Iniciando Kriging via GPU ({device}) | MODO: {mode.upper()}")
    
    biomes_to_run = [biome_input] if biome_input else BIOMES
    
    for biome in biomes_to_run:
        out_path = os.path.join(OUT_DIR, f"kriging_{biome}_{mode}_metrics.csv")
        if os.path.exists(out_path):
            print(f"[SKIP] Biome {biome.upper()} ({mode.upper()}) already exists in: {out_path}")
            continue

        print(f"\n[KERNEL] === PROCESSANDO BIOMA REAL: {biome.upper()} ===")
        pt_file = os.path.join(DATA_DIR, f"{biome}_v18.3_{PART}.pt")
        if not os.path.exists(pt_file):
            print(f"[ERROR] File {pt_file} not found.")
            continue

        print(f"[KERNEL] Carregando {pt_file}...")
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        
        l_pos = data['lidar'].pos
        l_z = data['lidar'].x[:, 0]
        d_pos = data['dem'].pos
        d_y = data['dem'].y
        
        legacy = LEGACY_STATS.get(biome, LEGACY_STATS['amazonia'])
        official = GNN_STATS.get(biome, GNN_STATS['default'])
        o_std, o_mean = legacy['std'], legacy['mean']
        n_std, n_mean = official['elev_std'], official['elev_mean']
        
        print(f"[KERNEL] Migrando Escala: Legacy({o_mean}/{o_std}) -> Official({n_mean}/{n_std})")
        l_z = (l_z * o_std + o_mean - n_mean) / n_std
        d_y[:, 0] = (d_y[:, 0] * o_std + o_mean - n_mean) / n_std
        
        del data
        gc.collect()
        
        print(f"[KERNEL] Ajustando Variograma (CPU)...")
        sub_idx = np.random.choice(len(l_pos), min(10000, len(l_pos)), replace=False)
        v_ok = OrdinaryKriging(
            l_pos[sub_idx, 0].numpy(), l_pos[sub_idx, 1].numpy(), l_z[sub_idx].numpy(),
            variogram_model='spherical', verbose=False, enable_plotting=False
        )
        v_params = v_ok.variogram_model_parameters.tolist()
        print(f"[KERNEL] Variograma ajustado: {v_params}")
        del v_ok
        
        l_pos_gpu = l_pos.to(device)
        l_z_gpu = l_z.to(device)
        d_pos_gpu = d_pos.to(device)
        d_y_gpu = d_y.to(device)
        
        min_x, max_x = d_pos[:, 0].min().item(), d_pos[:, 0].max().item()
        min_y, max_y = d_pos[:, 1].min().item(), d_pos[:, 1].max().item()
        
        if mode == 'golden':
            print(f"[KERNEL] Identifying the golden sector (4x4) for {biome.upper()}...")
            nx, ny = 4, 4
            dx = (max_x - min_x) / nx
            dy = (max_y - min_y) / ny
            max_lidar = -1
            best_box = None
            
            for ix in range(nx):
                for iy in range(ny):
                    bx0, bx1 = min_x + ix*dx, min_x + (ix+1)*dx
                    by0, by1 = min_y + iy*dy, min_y + (iy+1)*dy
                    count = ((l_pos[:,0] >= bx0) & (l_pos[:,0] < bx1) & (l_pos[:,1] >= by0) & (l_pos[:,1] < by1)).sum().item()
                    if count > max_lidar:
                        max_lidar = count
                        best_box = (bx0, bx1, by0, by1)
            
            print(f"[KERNEL] Setor de Ouro: Lon [{best_box[0]:.2f}, {best_box[1]:.2f}] | Lat [{best_box[2]:.2f}, {best_box[3]:.2f}] | LiDAR: {max_lidar}")
            min_x, max_x, min_y, max_y = best_box
            
        x_edges = np.linspace(min_x, max_x, 13)
        y_edges = np.linspace(min_y, max_y, 11)
        
        biome_results = []
        t_start_biome = time.time()
        
        
        for t_idx in range(start_tile, end_tile):
            if t_idx >= 120: break
            row = t_idx // 12
            col = t_idx % 12
            
            t_min_x, t_max_x = x_edges[col], x_edges[col+1]
            t_min_y, t_max_y = y_edges[row], y_edges[row+1]
            
            mask_dem = (d_pos_gpu[:, 0] >= t_min_x) & (d_pos_gpu[:, 0] <= t_max_x) & \
                       (d_pos_gpu[:, 1] >= t_min_y) & (d_pos_gpu[:, 1] <= t_max_y)
            
            if mask_dem.sum() == 0: continue
            
            buffer = 2000.0 
            mask_lidar = (l_pos_gpu[:, 0] >= t_min_x - buffer) & (l_pos_gpu[:, 0] <= t_max_x + buffer) & \
                         (l_pos_gpu[:, 1] >= t_min_y - buffer) & (l_pos_gpu[:, 1] <= t_max_y + buffer)
            
            l_pos_tile = l_pos_gpu[mask_lidar]
            l_z_tile = l_z_gpu[mask_lidar]
            d_pos_tile = d_pos_gpu[mask_dem]
            y_true_tile = d_y_gpu[mask_dem]
            
            if len(l_pos_tile) < 2: continue

            try:
                p_tile = torch_kriging_solve(l_pos_tile, l_z_tile, d_pos_tile, v_params)
                
                b_min, b_max = -50.0, 3000.0
                n_min = (b_min - n_mean) / n_std
                n_max = (b_max - n_mean) / n_std
                p_tile = p_tile.clamp(min=n_min, max=n_max)
                
                num_pts = d_pos_tile.shape[0]
                min_dist = torch.full((num_pts,), 1000.0, device=device)
                dist_batch = 4096
                
                for b_idx in range(0, num_pts, dist_batch):
                    end_idx = min(b_idx + dist_batch, num_pts)
                    d_sub = torch.cdist(d_pos_tile[b_idx:end_idx], l_pos_tile)
                    min_dist[b_idx:end_idx] = torch.min(d_sub, dim=1)[0]
                    del d_sub
                
                is_on = min_dist < 0.01
                is_off = ~is_on
                
                elev_std = GNN_STATS[biome]['elev_std']
                
                y_true_elev = y_true_tile[:, 0]
                y_true_slope = y_true_tile[:, 1]
                
                diff_m = torch.abs(y_true_elev - p_tile) * elev_std 
                
                is_steep = y_true_slope > 0.3
                is_gentle = y_true_slope < 0.05
                
                mae_all = diff_m.mean().item()
                mae_on = diff_m[is_on].mean().item() if is_on.any() else 1e-6
                mae_off = diff_m[is_off].mean().item() if is_off.any() else mae_all
                gap_ratio = mae_off / mae_on
                
                p90_all = torch.quantile(diff_m, 0.9).item()
                p90_steep = torch.quantile(diff_m[is_steep], 0.9).item() if is_steep.any() else p90_all
                p90_gentle = torch.quantile(diff_m[is_gentle], 0.9).item() if is_gentle.any() else p90_all

                results_tile = {
                    "biome": biome, 
                    "tile_idx": t_idx, 
                    "mae_m": mae_off,
                    "elevation_mae": mae_all,
                    "mae_lidar": mae_on,
                    "mae_no_lidar": mae_off,
                    "gap_ratio": gap_ratio,
                    "elevation_rmse": torch.sqrt((diff_m**2).mean()).item(),
                    "error_p90": p90_all,
                    "p90_steep_slope": p90_steep,
                    "p90_gentle_slope": p90_gentle
                }
                
                print(f"  [GPU] {biome.upper()} | Tile {t_idx:03d} | MAE_Fair: {mae_off:.3f}m | Gap: {gap_ratio:.2f}")
                biome_results.append(results_tile)

            except Exception as e:
                print(f"  [KERNEL] Failure in tile {t_idx}: {e}")
            
            if t_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                if biome_results:
                    pd.DataFrame(biome_results).to_csv(os.path.join(OUT_DIR, f"kriging_{biome}_fair_metrics.csv"), index=False)
                    print(f"  [CHECKPOINT] Partial metrics saved up to tile {t_idx}")
        
        if biome_results:
            df_res = pd.DataFrame(biome_results)
            df_res.to_csv(out_path, index=False)
            print(f"\n" + "="*50)
            print(f"RESUMO FINAL - {biome.upper()} ({mode.upper()})")
            print(f"Salvo em: {final_out}")
            print(f"MAE OFF-LiDAR (Fair): {df_res['mae_no_lidar'].mean():.4f}m")
            print(f"MAE ON-LiDAR (Anchor): {df_res['mae_lidar'].mean():.4f}m")
            print(f"Mean gap ratio:     {df_res['gap_ratio'].mean():.4f}")
            print(f"Mean RMSE (fair):   {df_res['elevation_rmse'].mean():.4f}m")
            print(f"Mean P90 (fair):    {df_res['error_p90'].mean():.4f}m")
            print("="*50)
            print(f"[KERNEL] Biome {biome} finished in {(time.time()-t_start_biome)/60:.1f} min.")
        
        del l_pos_gpu, l_z_gpu, d_pos_gpu, d_y_gpu
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--biome", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=['full', 'golden'], default='full')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=120)
    args = parser.parse_args()
    
    run_kriging_integrated(args.biome, args.mode, args.start, args.end)
