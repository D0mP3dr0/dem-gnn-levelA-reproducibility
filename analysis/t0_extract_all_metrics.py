INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime

BASE = Path(f"{INTERNAL_DATA_ROOT}")
ARTICLE = BASE / "internal" / "internal" / "internal"
OUT = ARTICLE / "resultados_expandidos"
INTERNAL_EXPORT_DIR = ARTICLE / "internal"
OUT.mkdir(exist_ok=True)
INTERNAL_EXPORT_DIR.mkdir(exist_ok=True)

GNN_PATHS = {
    "amazonia":       BASE / "v21/GNN_RUN_COLAB/results/checkpoints/amazonia/lightning_logs/version_0/training_log.csv",
    "cerrado":        BASE / "v21/GNN_RUN_COLAB/results/checkpoints/cerrado/lightning_logs/version_0/training_log.csv",
    "mata_atlantica": BASE / "v21/GNN_RUN_COLAB/results/checkpoints/mata_atlantica/version_0/training_log.csv",
    "pantanal":       BASE / "v21/GNN_RUN_COLAB/results/checkpoints/pantanal/lightning_logs/version_0/training_log.csv",
}

CNN_PATHS = {
    "amazonia":       BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/amazonia/version_1/metrics.csv",
    "cerrado":        BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/cerrado/version_1/metrics.csv",
    "mata_atlantica": BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/mata_atlantica/version_1/metrics.csv",
    "pantanal":       BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/pantanal/version_1/metrics.csv",
}

TILES_PATH = BASE / "testes_estatisticos/mae_per_tile.csv"
WILCOXON_PATH = BASE / "testes_estatisticos/wilcoxon_results.csv"
BOOTSTRAP_PATH = BASE / "testes_estatisticos/bootstrap_results.csv"

GNN_HPARAMS = {
    "amazonia":       BASE / "v21/GNN_RUN_COLAB/results/checkpoints/amazonia/lightning_logs/version_0/hparams.yaml",
    "cerrado":        BASE / "v21/GNN_RUN_COLAB/results/checkpoints/cerrado/lightning_logs/version_0/hparams.yaml",
    "mata_atlantica": BASE / "v21/GNN_RUN_COLAB/results/checkpoints/mata_atlantica/version_0/hparams.yaml",
    "pantanal":       BASE / "v21/GNN_RUN_COLAB/results/checkpoints/pantanal/lightning_logs/version_0/hparams.yaml",
}

CNN_HPARAMS = {
    "amazonia":       BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/amazonia/version_1/hparams.yaml",
    "cerrado":        BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/cerrado/version_1/hparams.yaml",
    "mata_atlantica": BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/mata_atlantica/version_1/hparams.yaml",
    "pantanal":       BASE / "CNN_U-NET/ResU-Net_RUN_COLAB/results/pantanal/version_1/hparams.yaml",
}


def extract_gnn_best_epoch(path, biome):
    df = pd.read_csv(path)
    
    max_phase = df["phase"].max()
    df_last_phase = df[df["phase"] == max_phase].copy()
    
    if df_last_phase.empty:
        df_last_phase = df.copy()
    
    best_idx = df_last_phase["elevation_mae"].idxmin()
    best_row = df.loc[best_idx]
    
    result = {"biome": biome, "model": "GNN", "source_file": str(path)}
    result["selection_criterion"] = f"min(elevation_mae) at phase={max_phase}"
    result["best_epoch"] = int(best_row["epoch"])
    result["best_phase"] = int(best_row["phase"])
    
    for col in df.columns:
        result[col] = best_row[col]
    
    return result, df


def extract_cnn_best_epoch(path, biome):
    df = pd.read_csv(path)
    
    df_val = df.dropna(subset=["elevation_mae"]).copy()
    
    if df_val.empty:
        return None, df
    
    best_idx = df_val["elevation_mae"].idxmin()
    best_row = df_val.loc[best_idx]
    
    result = {"biome": biome, "model": "CNN", "source_file": str(path)}
    result["selection_criterion"] = "min(elevation_mae) among non-null rows"
    result["best_epoch"] = int(best_row["epoch"])
    result["best_phase"] = -1
    
    for col in df_val.columns:
        val = best_row[col]
        if pd.notna(val):
            result[col] = val
    
    return result, df


def build_reconciliation(gnn_bests, cnn_bests, tiles_df):
    rows = []
    for biome in ["amazonia", "cerrado", "mata_atlantica", "pantanal"]:
        gnn_b = next((r for r in gnn_bests if r["biome"] == biome), None)
        cnn_b = next((r for r in cnn_bests if r["biome"] == biome), None)
        
        gnn_tiles = tiles_df[(tiles_df["biome"] == biome) & (tiles_df["model"] == "GNN")]["mae_m"]
        cnn_tiles = tiles_df[(tiles_df["biome"] == biome) & (tiles_df["model"] == "CNN")]["mae_m"]
        
        row = {
            "biome": biome,
            "gnn_train_val_mae": gnn_b["elevation_mae"] if gnn_b else None,
            "gnn_train_val_rmse": gnn_b.get("elevation_rmse") if gnn_b else None,
            "gnn_train_best_epoch": gnn_b.get("best_epoch") if gnn_b else None,
            "gnn_train_best_phase": gnn_b.get("best_phase") if gnn_b else None,
            "gnn_inference_mae_mean": gnn_tiles.mean() if len(gnn_tiles) > 0 else None,
            "gnn_inference_mae_median": gnn_tiles.median() if len(gnn_tiles) > 0 else None,
            "gnn_inference_mae_std": gnn_tiles.std() if len(gnn_tiles) > 0 else None,
            "gnn_inference_n_tiles": len(gnn_tiles),
            "gnn_discrepancy_ratio": None,
            
            "cnn_train_val_mae": cnn_b["elevation_mae"] if cnn_b else None,
            "cnn_train_val_rmse": cnn_b.get("elevation_rmse") if cnn_b else None,
            "cnn_train_best_epoch": cnn_b.get("best_epoch") if cnn_b else None,
            "cnn_inference_mae_mean": cnn_tiles.mean() if len(cnn_tiles) > 0 else None,
            "cnn_inference_mae_median": cnn_tiles.median() if len(cnn_tiles) > 0 else None,
            "cnn_inference_mae_std": cnn_tiles.std() if len(cnn_tiles) > 0 else None,
            "cnn_inference_n_tiles": len(cnn_tiles),
            "cnn_discrepancy_ratio": None,
            
            "inference_winner": None,
            "inference_gnn_improvement_pct": None,
        }
        
        if row["gnn_train_val_mae"] and row["gnn_inference_mae_mean"]:
            row["gnn_discrepancy_ratio"] = row["gnn_inference_mae_mean"] / row["gnn_train_val_mae"]
        if row["cnn_train_val_mae"] and row["cnn_inference_mae_mean"]:
            row["cnn_discrepancy_ratio"] = row["cnn_inference_mae_mean"] / row["cnn_train_val_mae"]
        
        if row["gnn_inference_mae_mean"] is not None and row["cnn_inference_mae_mean"] is not None:
            if row["gnn_inference_mae_mean"] < row["cnn_inference_mae_mean"]:
                row["inference_winner"] = "GNN"
                row["inference_gnn_improvement_pct"] = (
                    (row["cnn_inference_mae_mean"] - row["gnn_inference_mae_mean"])
                    / row["cnn_inference_mae_mean"] * 100
                )
            else:
                row["inference_winner"] = "CNN"
                row["inference_gnn_improvement_pct"] = -(
                    (row["gnn_inference_mae_mean"] - row["cnn_inference_mae_mean"])
                    / row["cnn_inference_mae_mean"] * 100
                )
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def INTERNAL_copy_sources():
    gnn_dir = INTERNAL_EXPORT_DIR / "gnn_training_logs"
    cnn_dir = INTERNAL_EXPORT_DIR / "cnn_training_logs"
    stats_dir = INTERNAL_EXPORT_DIR / "statistical_tests"
    hp_dir = INTERNAL_EXPORT_DIR / "hparams"
    
    for d in [gnn_dir, cnn_dir, stats_dir, hp_dir]:
        d.mkdir(exist_ok=True)
    
    for biome, path in GNN_PATHS.items():
        if path.exists():
            shutil.copy2(path, gnn_dir / f"gnn_{biome}_training_log.csv")
    
    for biome, path in CNN_PATHS.items():
        if path.exists():
            shutil.copy2(path, cnn_dir / f"cnn_{biome}_metrics.csv")
    
    for biome, path in GNN_HPARAMS.items():
        if path.exists():
            shutil.copy2(path, hp_dir / f"gnn_{biome}_hparams.yaml")
    
    for biome, path in CNN_HPARAMS.items():
        if path.exists():
            shutil.copy2(path, hp_dir / f"cnn_{biome}_hparams.yaml")
    
    for src in [TILES_PATH, WILCOXON_PATH, BOOTSTRAP_PATH]:
        if src.exists():
            shutil.copy2(src, stats_dir / src.name)
    
    tiles_gnn = BASE / "testes_estatisticos/mae_per_tile_GNN_ONLY.csv"
    tiles_cnn = BASE / "testes_estatisticos/mae_per_tile_CNN_ONLY.csv"
    for src in [tiles_gnn, tiles_cnn]:
        if src.exists():
            shutil.copy2(src, stats_dir / src.name)


def main():
    print("=" * 70)
    print("T0.1 — STANDARDIZED METRICS EXTRACTION")
    print("=" * 70)
    
    gnn_bests = []
    gnn_timeseries = []
    
    for biome, path in GNN_PATHS.items():
        print(f"\n[GNN] {biome}: {path}")
        if not path.exists():
            print(f"  ERROR: file not found")
            continue
        best, df = extract_gnn_best_epoch(path, biome)
        gnn_bests.append(best)
        df_ts = df.copy()
        df_ts.insert(0, "biome", biome)
        df_ts.insert(1, "model", "GNN")
        gnn_timeseries.append(df_ts)
        print(f"  Best epoch={best['best_epoch']}, phase={best['best_phase']}")
        print(f"  elevation_mae={best['elevation_mae']:.4f}m, elevation_rmse={best['elevation_rmse']:.4f}m")
        print(f"  Extracted columns: {len([k for k in best.keys() if k not in ['biome','model','source_file','selection_criterion','best_epoch','best_phase']])}")
    
    cnn_bests = []
    cnn_timeseries = []
    
    for biome, path in CNN_PATHS.items():
        print(f"\n[CNN] {biome}: {path}")
        if not path.exists():
            print(f"  ERROR: file not found")
            continue
        best, df = extract_cnn_best_epoch(path, biome)
        if best is None:
            print(f"  ERROR: no valid rows for elevation_mae")
            continue
        cnn_bests.append(best)
        df_ts = df.copy()
        df_ts.insert(0, "biome", biome)
        df_ts.insert(1, "model", "CNN")
        cnn_timeseries.append(df_ts)
        print(f"  Best epoch={best['best_epoch']}")
        print(f"  elevation_mae={best['elevation_mae']:.4f}m, elevation_rmse={best.get('elevation_rmse', 'N/A')}")
        print(f"  Extracted columns: {len([k for k in best.keys() if k not in ['biome','model','source_file','selection_criterion','best_epoch','best_phase']])}")
    
    gnn_best_df = pd.DataFrame(gnn_bests)
    cnn_best_df = pd.DataFrame(cnn_bests)
    
    gnn_best_df.to_csv(OUT / "gnn_best_epoch_all_metrics.csv", index=False)
    cnn_best_df.to_csv(OUT / "cnn_best_epoch_all_metrics.csv", index=False)
    print(f"\n  Saved: gnn_best_epoch_all_metrics.csv ({len(gnn_best_df.columns)} columns)")
    print(f"  Saved: cnn_best_epoch_all_metrics.csv ({len(cnn_best_df.columns)} columns)")
    
    if gnn_timeseries:
        gnn_ts_df = pd.concat(gnn_timeseries, ignore_index=True)
        gnn_ts_df.to_csv(OUT / "gnn_all_epochs_timeseries.csv", index=False)
        print(f"  Saved: gnn_all_epochs_timeseries.csv ({len(gnn_ts_df)} rows)")
    
    if cnn_timeseries:
        cnn_ts_df = pd.concat(cnn_timeseries, ignore_index=True)
        cnn_ts_df.to_csv(OUT / "cnn_all_epochs_timeseries.csv", index=False)
        print(f"  Saved: cnn_all_epochs_timeseries.csv ({len(cnn_ts_df)} rows)")
    
    print("\n" + "=" * 70)
    print("T0.2 — TRAINING vs INFERENCE RECONCILIATION")
    print("=" * 70)
    
    tiles_df = pd.read_csv(TILES_PATH)
    recon = build_reconciliation(gnn_bests, cnn_bests, tiles_df)
    recon.to_csv(OUT / "reconciliation_train_vs_inference.csv", index=False)
    
    print(f"\n{'Biome':<18} {'GNN val MAE':>12} {'GNN inf MAE':>12} {'ratio':>8} | {'CNN val MAE':>12} {'CNN inf MAE':>12} {'ratio':>8} | {'Winner':>6}")
    print("-" * 110)
    for _, r in recon.iterrows():
        gnn_r = f"{r['gnn_discrepancy_ratio']:.2f}x" if pd.notna(r['gnn_discrepancy_ratio']) else "N/A"
        cnn_r = f"{r['cnn_discrepancy_ratio']:.2f}x" if pd.notna(r['cnn_discrepancy_ratio']) else "N/A"
        print(f"{r['biome']:<18} {r['gnn_train_val_mae']:>12.3f} {r['gnn_inference_mae_mean']:>12.3f} {gnn_r:>8} | {r['cnn_train_val_mae']:>12.3f} {r['cnn_inference_mae_mean']:>12.3f} {cnn_r:>8} | {r['inference_winner']:>6}")
    
    print(f"\n  Saved: reconciliation_train_vs_inference.csv")
    
    print("\n" + "=" * 70)
    print("COPYING SOURCE RECORDS TO FOLDER")
    print("=" * 70)
    INTERNAL_copy_sources()
    print("  Data copied to internal/")
    
    provenance = {
        "generated_at": datetime.now().isoformat(),
        "script": "t0_extract_all_metrics.py",
        "gnn_sources": {b: str(p) for b, p in GNN_PATHS.items()},
        "cnn_sources": {b: str(p) for b, p in CNN_PATHS.items()},
        "inference_source": str(TILES_PATH),
        "statistical_sources": {
            "wilcoxon": str(WILCOXON_PATH),
            "bootstrap": str(BOOTSTRAP_PATH),
        },
        "note": "Inference tiles generated from phase_5 checkpoints of the respective biomes"
    }
    
    with open(OUT / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    with open(INTERNAL_EXPORT_DIR / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    
    print(f"\n  Saved: provenance.json")
    print("\n" + "=" * 70)
    print("T0.1 + T0.2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
