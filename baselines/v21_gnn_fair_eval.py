INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
INTERNAL_GRAPH_ROOT = __import__("os").environ.get("DEMGNN_GRAPH_ROOT", ".")
import os
import sys
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

GNN_ROOT = Path(f"{INTERNAL_DATA_ROOT}/v21/GNN_RUN_COLAB").absolute()
sys.path.insert(0, str(GNN_ROOT))
sys.path.insert(0, str(GNN_ROOT / "training"))

os.chdir(str(GNN_ROOT))

from training.module_v18 import HierarchicalTrainerModule
from config import NORMALIZATION_STATS

def run_fair_evaluation(biome, checkpoint_path, graph_path):
    print(f"\n[GNN EVAL] Starting evaluation for: {biome.upper()}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Grafo: {graph_path}")

    if not graph_path.exists():
        print(f"  [ERROR] Graph not found in {graph_path}")
        return None

    data = torch.load(str(graph_path), map_location="cpu", weights_only=False)
    metadata = data.metadata()
    
    model = HierarchicalTrainerModule.load_from_checkpoint(
        checkpoint_path,
        metadata=metadata,
        map_location="cpu"
    )
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)

    from torch_geometric.loader import NeighborLoader
    
    print(f"  [AUDIT] Analysing LiDAR density for the spatial crop...")
    coords = data['dem'].pos if hasattr(data['dem'], 'pos') else data['dem'].x[:, 10:12]
    lidar_mask = data['dem'].x[:, 14].bool()
    
    min_x, min_y = coords.min(dim=0)[0]
    max_x, max_y = coords.max(dim=0)[0]
    
    nx, ny = 4, 4 
    dx = (max_x - min_x) / nx
    dy = (max_y - min_y) / ny
    
    best_sector_nodes = None
    max_lidar_count = -1
    
    for ix in range(nx):
        for iy in range(ny):
            x0, x1 = min_x + ix*dx, min_x + (ix+1)*dx
            y0, y1 = min_y + iy*dy, min_y + (iy+1)*dy
            
            in_sector = (coords[:, 0] >= x0) & (coords[:, 0] < x1) & \
                        (coords[:, 1] >= y0) & (coords[:, 1] < y1)
            
            count = lidar_mask[in_sector].sum().item()
            if count > max_lidar_count:
                max_lidar_count = count
                best_sector_nodes = torch.where(in_sector)[0]

    num_nodes_final = len(best_sector_nodes)
    print(f"  [AUDIT] Golden sector selected: {num_nodes_final:,} nodes | LiDAR: {max_lidar_count:,} points.")
    
    min_vals = coords[best_sector_nodes].min(dim=0)[0]
    max_vals = coords[best_sector_nodes].max(dim=0)[0]
    s_min_x, s_min_y = min_vals[0].item(), min_vals[1].item()
    s_max_x, s_max_y = max_vals[0].item(), max_vals[1].item()
    print(f"  [AUDIT] Bounding Box do Setor: Lon [{s_min_x:.4f}, {s_max_x:.4f}] | Lat [{s_min_y:.4f}, {s_max_y:.4f}]")
    
    input_nodes = ('dem', best_sector_nodes)
    
    num_neighbors = {et: [10] * 6 for et in data.edge_types}
    
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=1536, 
        shuffle=False,
        num_workers=0
    )

    print(f"  [EVAL] Processing {num_nodes_final:,} nodes (optimized sector) in batches...")
    
    all_preds = []
    all_targets = []
    all_has_lidar = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            out = model.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            
            batch_size_actual = batch['dem'].batch_size
            
            p_batch = out['elevation'][:batch_size_actual].view(-1)
            t_batch = batch['dem'].y[:batch_size_actual, 0].view(-1)
            h_batch = batch['dem'].x[:batch_size_actual, 14].bool()
            
            all_preds.append(p_batch.cpu())
            all_targets.append(t_batch.cpu())
            all_has_lidar.append(h_batch.cpu())

            if i % 10 == 0:
                print(f"    - [{biome.upper()}] Progresso: Batch {i}/{len(loader)} ({(i/len(loader))*100:.1f}%)")

    p_val = torch.cat(all_preds)
    t_val = torch.cat(all_targets)
    has_lidar = torch.cat(all_has_lidar)

    stats = NORMALIZATION_STATS.get(biome, NORMALIZATION_STATS['default'])
    std = stats['elev_std']
    mean = stats['elev_mean']
    
    p_val_m = p_val * std + mean
    t_val_m = t_val * std + mean
    
    abs_err = torch.abs(p_val_m - t_val_m)
    
    mae_anchor = abs_err[has_lidar].mean().item() if has_lidar.any() else 0.0
    
    mae_gap = abs_err[~has_lidar].mean().item() if (~has_lidar).any() else 0.0
    
    mae_global = abs_err.mean().item()
    rmse = (abs_err ** 2).mean().sqrt().item()
    p90 = abs_err.quantile(0.9).item()
    gap_ratio = mae_gap / mae_anchor if mae_anchor > 1e-9 else 0.0

    print(f"  [RESULT] MAE Global: {mae_global:.3f}m")
    print(f"  [RESULT] MAE Anchor: {mae_anchor:.3f}m")
    print(f"  [RESULT] MAE Gap:    {mae_gap:.3f}m")
    print(f"  [RESULT] Gap Ratio:  {gap_ratio:.3f}")
    print(f"  [RESULT] RMSE:       {rmse:.3f}m")
    print(f"  [RESULT] P90:        {p90:.3f}m")

    return {
        "biome":      biome,
        "mae_m":      mae_global,
        "mae_anchor": mae_anchor,
        "mae_gap":    mae_gap,
        "gap_ratio":  gap_ratio,
        "rmse_m":     rmse,
        "p90_m":      p90,
        "n_nodes":    len(p_val),
        "n_anchor":   int(has_lidar.sum().item()),
        "n_gap":      int((~has_lidar).sum().item()),
    }

def main(only_biome=None, append=False):
    ckpt_base = Path(f"{INTERNAL_DATA_ROOT}/v21/GNN_RUN_COLAB/results/checkpoints")

    checkpoints = {
        "amazonia":       ckpt_base / "amazonia/phase_3/amazonia_p3_epoch=03_val_loss=0.0230.ckpt",
        "cerrado":        ckpt_base / "cerrado/cerrado/phase_5/cerrado_p5_epoch=03_val_loss=0.0179.ckpt",
        "mata_atlantica": ckpt_base / "mata_atlantica/phase_3/mata_atlantica_p3_epoch=13_val_loss=0.0250.ckpt",
        "pantanal":       ckpt_base / "pantanal/phase_5/pantanal_p5_epoch=03_val_loss=0.0239.ckpt",
    }

    graph_base = Path(f"{INTERNAL_GRAPH_ROOT}/GINE_GRAPH")

    output_path_1 = Path(f"{INTERNAL_DATA_ROOT}/kriging/results/gnn_official_fair_metrics.csv")
    output_path_2 = Path(
        f"{INTERNAL_DATA_ROOT}/internal/internal/baselines/gnn_official_fair_metrics.csv"
    )

    if only_biome:
        if only_biome not in checkpoints:
            print(f"[ERROR] Biome '{only_biome}' not recognised. Options: {list(checkpoints.keys())}")
            return
        checkpoints = {only_biome: checkpoints[only_biome]}

    print("[PRE-CHECK] Verificando checkpoints...")
    for biome, ckpt in checkpoints.items():
        status = "OK" if ckpt.exists() else "FALTANDO"
        print(f"  [{status}] {biome}: {ckpt.name}")
    print()

    results = []
    if append and output_path_1.exists():
        existing = pd.read_csv(output_path_1)
        results = existing.to_dict("records")
        biomes_ja_feitos = {r["biome"] for r in results}
        print(f"[APPEND] {len(results)} biomas ja presentes: {biomes_ja_feitos}")

    for biome, ckpt in checkpoints.items():
        if not ckpt.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt}")
            continue
        graph_path = graph_base / f"{biome}_v18.3_part2.pt"
        res = run_fair_evaluation(biome, ckpt, graph_path)
        if res:
            res["eval_source"] = "golden_sector_part2_inference"
            res["checkpoint"] = ckpt.name
            results = [r for r in results if r["biome"] != biome]
            results.append(res)
            df = pd.DataFrame(results)
            df.to_csv(output_path_1, index=False)
            df.to_csv(output_path_2, index=False)
            print(f"  [SAVE] {biome.upper()} salvo incrementalmente.")

    print(f"\n[DONE] GNN evaluation finished. {len(results)} biomes in the CSV.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--biome", type=str, default=None,
                        help='Run only one biome (amazonia|cerrado|mata_atlantica|pantanal)')
    parser.add_argument("--append", action="store_true",
                        help='Append to the existing CSV instead of overwriting')
    args = parser.parse_args()
    main(only_biome=args.biome, append=args.append)
