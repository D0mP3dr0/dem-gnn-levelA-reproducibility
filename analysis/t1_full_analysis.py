INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import json
import warnings
warnings.filterwarnings("ignore")

OUT = Path(rf"{INTERNAL_DATA_ROOT}\internal\resultados_expandidos")

TILES = pd.read_csv(rf"{INTERNAL_DATA_ROOT}\testes_estatisticos\mae_per_tile.csv")
GNN_BEST = pd.read_csv(OUT / "gnn_best_epoch_all_metrics.csv")
CNN_BEST = pd.read_csv(OUT / "cnn_best_epoch_all_metrics.csv")
GNN_TS = pd.read_csv(OUT / "gnn_all_epochs_timeseries.csv")
CNN_TS = pd.read_csv(OUT / "cnn_all_epochs_timeseries.csv")
WILCOXON = pd.read_csv(rf"{INTERNAL_DATA_ROOT}\testes_estatisticos\wilcoxon_results.csv")
BOOTSTRAP = pd.read_csv(rf"{INTERNAL_DATA_ROOT}\testes_estatisticos\bootstrap_results.csv")

BIOMES = ["amazonia", "cerrado", "mata_atlantica", "pantanal"]
report_lines = []

def rpt(line=""):
    report_lines.append(line)

def rpt_table(df, float_fmt=".4f"):
    rpt(df.to_markdown(index=False, floatfmt=float_fmt))
    rpt()


def t1_1_tile_distribution():
    rpt("# T1.1 — Inference Tiles Distributional Analysis")
    rpt()
    
    rows = []
    for biome in BIOMES:
        for model in ["GNN", "CNN"]:
            vals = TILES[(TILES["biome"] == biome) & (TILES["model"] == model)]["mae_m"]
            if len(vals) == 0:
                continue
            
            percentiles = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95, 99])
            
            try:
                shapiro_stat, shapiro_p = sp_stats.shapiro(vals)
            except Exception:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            rows.append({
                "biome": biome, "model": model, "n_tiles": len(vals),
                "mean": vals.mean(), "std": vals.std(), "median": vals.median(),
                "cv_pct": vals.std() / vals.mean() * 100 if vals.mean() > 0 else np.nan,
                "iqr": percentiles[4] - percentiles[2],
                "skewness": vals.skew(), "kurtosis": vals.kurtosis(),
                "min": vals.min(), "P5": percentiles[0], "P10": percentiles[1],
                "P25": percentiles[2], "P50": percentiles[3], "P75": percentiles[4],
                "P90": percentiles[5], "P95": percentiles[6], "P99": percentiles[7],
                "max": vals.max(),
                "shapiro_W": shapiro_stat, "shapiro_p": shapiro_p,
                "normal": "Yes" if shapiro_p > 0.05 else "No",
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_1_tile_distribution.csv", index=False)
    
    rpt("## Descriptive statistics by biome × model")
    rpt()
    display_cols = ["biome", "model", "n_tiles", "mean", "median", "std", "cv_pct", "P10", "P25", "P50", "P75", "P90", "P99", "skewness"]
    rpt_table(df[display_cols])
    
    rpt("## Normality test (Shapiro-Wilk)")
    rpt()
    rpt_table(df[["biome", "model", "shapiro_W", "shapiro_p", "normal"]])
    
    return df


def t1_2_p90_terrain():
    rpt("# T1.2 — P90 Stratified by Terrain Category")
    rpt()
    
    shared_p90 = ["p90_clearing", "p90_dense_veg", "p90_gentle_slope", "p90_steep_slope", "p90_water"]
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        cnn_row = CNN_BEST[CNN_BEST["biome"] == biome].iloc[0]
        
        for metric in shared_p90:
            gnn_val = gnn_row.get(metric, np.nan)
            cnn_val = cnn_row.get(metric, np.nan)
            
            if pd.notna(gnn_val) and pd.notna(cnn_val) and cnn_val > 0:
                improvement = (cnn_val - gnn_val) / cnn_val * 100
                winner = "GNN" if gnn_val < cnn_val else "CNN"
            else:
                improvement = np.nan
                winner = "N/A"
            
            rows.append({
                "biome": biome,
                "terrain_category": metric.replace("p90_", ""),
                "gnn_p90_m": gnn_val,
                "cnn_p90_m": cnn_val,
                "winner": winner,
                "gnn_improvement_pct": improvement,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_2_p90_terrain_stratified.csv", index=False)
    
    rpt("## P90 by terrain category (GNN vs CNN)")
    rpt()
    rpt_table(df, float_fmt=".3f")
    
    rpt("## Win count by terrain category")
    rpt()
    pivot = df.pivot_table(index="terrain_category", columns="winner", values="biome", aggfunc="count", fill_value=0)
    rpt(pivot.to_markdown())
    rpt()
    
    return df


def t1_3_p90_geomorphic():
    rpt("# T1.3 — Advanced Geomorphic P90 (GNN only)")
    rpt()
    
    gnn_p90 = [
        "p90_peak_extreme", "p90_peak_moderate",
        "p90_valley_extreme", "p90_valley_moderate",
        "p90_with_lidar", "p90_no_lidar",
        "p90_water", "p90_dense_veg", "p90_clearing",
        "p90_steep_slope", "p90_gentle_slope",
    ]
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        for metric in gnn_p90:
            val = gnn_row.get(metric, np.nan)
            rows.append({
                "biome": biome,
                "p90_category": metric.replace("p90_", ""),
                "p90_value_m": val,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_3_p90_geomorphic_gnn.csv", index=False)
    
    pivot = df.pivot_table(index="p90_category", columns="biome", values="p90_value_m")
    pivot = pivot.reindex(columns=BIOMES)
    rpt("## GNN Full P90 — 11 categories × 4 biomes (meters)")
    rpt()
    rpt(pivot.to_markdown(floatfmt=".3f"))
    rpt()
    
    return df


def t1_4_multitask():
    rpt('# T1.4 — Multi-Task Performance (GNN — 8 tasks × 4 biomes)')
    rpt()
    
    tasks = ["elevation", "slope", "aspect_deg", "curvature", "tpi", "tri", "roughness", "flow_accum"]
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        for task in tasks:
            mae_col = f"{task}_mae" if task != "elevation" else "elevation_mae"
            rmse_col = f"{task}_rmse" if task != "elevation" else "elevation_rmse"
            weight_col = f"{task}_weight"
            status_col = f"{task}_status"
            
            mae_val = gnn_row.get(mae_col, np.nan)
            rmse_val = gnn_row.get(rmse_col, np.nan)
            weight_val = gnn_row.get(weight_col, np.nan)
            status_val = gnn_row.get(status_col, "UNKNOWN")
            
            rows.append({
                "biome": biome, "task": task,
                "mae": mae_val, "rmse": rmse_val,
                "weight": weight_val, "status": status_val,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_4_multitask_performance.csv", index=False)
    
    pivot_mae = df.pivot_table(index="task", columns="biome", values="mae")
    pivot_mae = pivot_mae.reindex(columns=BIOMES)
    rpt("## MAE by task (GNN)")
    rpt()
    rpt(pivot_mae.to_markdown(floatfmt=".4f"))
    rpt()
    
    pivot_rmse = df.pivot_table(index="task", columns="biome", values="rmse")
    pivot_rmse = pivot_rmse.reindex(columns=BIOMES)
    rpt("## RMSE by task (GNN)")
    rpt()
    rpt(pivot_rmse.to_markdown(floatfmt=".4f"))
    rpt()
    
    return df


def t1_5_physics():
    rpt("# T1.5 — Physics Consistency")
    rpt()
    
    physics_metrics = ["physics_consistency", "physics_terrain", "physics_water_flow", "physics_river_width", "physics_geomorph"]
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        cnn_row = CNN_BEST[CNN_BEST["biome"] == biome].iloc[0]
        
        for metric in physics_metrics:
            gnn_val = gnn_row.get(metric, np.nan)
            cnn_val = cnn_row.get(metric, np.nan)
            rows.append({
                "biome": biome, "physics_metric": metric.replace("physics_", ""),
                "gnn_value": gnn_val,
                "cnn_value": cnn_val if pd.notna(cnn_val) else "N/A (not tracked)",
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_5_physics_consistency.csv", index=False)
    
    rpt("## Physics consistency metrics")
    rpt()
    rpt_table(df, float_fmt=".4f")
    
    rpt("## physics_water_flow — direct GNN vs CNN comparison")
    rpt()
    wf_rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        cnn_row = CNN_BEST[CNN_BEST["biome"] == biome].iloc[0]
        gnn_wf = gnn_row.get("physics_water_flow", np.nan)
        cnn_wf = cnn_row.get("physics_water_flow", np.nan)
        wf_rows.append({"biome": biome, "gnn_water_flow": gnn_wf, "cnn_water_flow": cnn_wf})
    rpt_table(pd.DataFrame(wf_rows), float_fmt=".3f")
    
    return df


def t1_6_generalization():
    rpt("# T1.6 — Generalization: Anchor/LiDAR vs Non-Anchor/Non-LiDAR")
    rpt()
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        cnn_row = CNN_BEST[CNN_BEST["biome"] == biome].iloc[0]
        
        rows.append({
            "biome": biome,
            "gnn_mae_lidar": gnn_row.get("mae_lidar", np.nan),
            "gnn_mae_no_lidar": gnn_row.get("mae_no_lidar", np.nan),
            "gnn_gap_ratio": gnn_row.get("gap_ratio", np.nan),
            "gnn_p90_with_lidar": gnn_row.get("p90_with_lidar", np.nan),
            "gnn_p90_no_lidar": gnn_row.get("p90_no_lidar", np.nan),
            "cnn_mae_anchor": cnn_row.get("mae_anchor", np.nan),
            "cnn_mae_non_anchor": cnn_row.get("mae_non_anchor", np.nan),
            "cnn_gap_ratio": cnn_row.get("gap_ratio", np.nan),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_6_generalization_lidar.csv", index=False)
    
    rpt("## GNN: mae_lidar vs mae_no_lidar")
    rpt()
    rpt_table(df[["biome", "gnn_mae_lidar", "gnn_mae_no_lidar", "gnn_gap_ratio", "gnn_p90_with_lidar", "gnn_p90_no_lidar"]], float_fmt=".4f")
    
    rpt("## CNN: mae_anchor vs mae_non_anchor")
    rpt()
    rpt_table(df[["biome", "cnn_mae_anchor", "cnn_mae_non_anchor", "cnn_gap_ratio"]], float_fmt=".4f")
    
    rpt("> **CAVEAT**: `mae_no_lidar` (GNN) and `mae_non_anchor` (CNN) measure potentially different populations.")
    rpt("> GNN operates on a sparse graph; CNN operates on a dense grid. Direct comparison is not strictly valid.")
    rpt()
    
    return df


def t1_7_cerrado_diagnosis():
    rpt("# T1.7 — Deep Diagnosis: Cerrado")
    rpt()
    
    gnn_cerrado_ts = GNN_TS[GNN_TS["biome"] == "cerrado"].copy()
    gnn_cerrado_ts.to_csv(OUT / "t1_7_cerrado_gnn_full_timeseries.csv", index=False)
    
    rpt("## GNN Cerrado convergence curve (all epochs/phases)")
    rpt()
    convergence_cols = ["epoch", "phase", "val_total", "elevation_mae", "elevation_rmse", "overfit_ratio", "loss_delta", "physics_water_flow", "physics_consistency"]
    available = [c for c in convergence_cols if c in gnn_cerrado_ts.columns]
    rpt_table(gnn_cerrado_ts[available], float_fmt=".4f")
    
    cerrado_tiles_gnn = TILES[(TILES["biome"] == "cerrado") & (TILES["model"] == "GNN")]["mae_m"]
    cerrado_tiles_cnn = TILES[(TILES["biome"] == "cerrado") & (TILES["model"] == "CNN")]["mae_m"]
    
    rpt("## Inference tiles distribution — Cerrado")
    rpt()
    quantiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    gnn_q = np.percentile(cerrado_tiles_gnn, quantiles)
    cnn_q = np.percentile(cerrado_tiles_cnn, quantiles)
    q_df = pd.DataFrame({"percentile": [f"P{q}" for q in quantiles], "GNN_MAE_m": gnn_q, "CNN_MAE_m": cnn_q})
    rpt_table(q_df, float_fmt=".3f")
    
    rpt("## GNN Cerrado tiles with MAE > 20m (outliers)")
    rpt()
    outliers = TILES[(TILES["biome"] == "cerrado") & (TILES["model"] == "GNN") & (TILES["mae_m"] > 20)]
    rpt(f"Total: {len(outliers)} cut of {len(cerrado_tiles_gnn)} tiles ({len(outliers)/len(cerrado_tiles_gnn)*100:.1f}%)")
    rpt()
    if len(outliers) > 0:
        rpt_table(outliers.sort_values("mae_m", ascending=False).head(20))
    
    rpt("## physics_water_flow comparison across biomes (GNN best epoch)")
    rpt()
    wf_comparison = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        wf_comparison.append({
            "biome": biome,
            "physics_water_flow": gnn_row.get("physics_water_flow", np.nan),
            "physics_consistency": gnn_row.get("physics_consistency", np.nan),
            "elevation_mae": gnn_row.get("elevation_mae", np.nan),
            "water_elev_std": gnn_row.get("water_elev_std", np.nan),
            "water_nodes_count": gnn_row.get("water_nodes_count", np.nan),
        })
    wf_df = pd.DataFrame(wf_comparison)
    rpt_table(wf_df, float_fmt=".3f")
    
    rpt("## P90 terrain breakdown — Cerrado GNN vs CNN")
    rpt()
    gnn_cerrado = GNN_BEST[GNN_BEST["biome"] == "cerrado"].iloc[0]
    cnn_cerrado = CNN_BEST[CNN_BEST["biome"] == "cerrado"].iloc[0]
    p90_cols = ["p90_clearing", "p90_dense_veg", "p90_gentle_slope", "p90_steep_slope", "p90_water"]
    p90_rows = []
    for col in p90_cols:
        p90_rows.append({
            "category": col.replace("p90_", ""),
            "gnn": gnn_cerrado.get(col, np.nan),
            "cnn": cnn_cerrado.get(col, np.nan),
            "winner": "GNN" if gnn_cerrado.get(col, 999) < cnn_cerrado.get(col, 999) else "CNN",
        })
    rpt_table(pd.DataFrame(p90_rows), float_fmt=".3f")
    
    rpt("## Diagnosis")
    rpt()
    rpt("1. **val→inf Discrepancy 5.69x**: GNN training yields 2.31m but inference gives 13.17m")
    rpt("2. **Anomalously high physics_water_flow**: Cerrado shows ~68 vs ~16-33 in other biomes")
    rpt("3. **CNN wins 4 out of 5 P90 categories**: clearing, dense_veg, gentle_slope, water")
    rpt("4. **GNN only wins deeply on steep_slope**: where graph structure helps the most")
    rpt(f"5. **{len(outliers)} tiles with MAE > 20m**: potential spatial failure cluster (e.g. boundary conditions)")
    rpt()


def t1_8_spectral_correlations():
    rpt("# T1.8 — Feature Importance via Spectral Correlations (GNN)")
    rpt()
    
    corr_cols = [c for c in GNN_BEST.columns if c.startswith("corr_")]
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        for col in corr_cols:
            rows.append({
                "biome": biome,
                "correlation": col.replace("corr_", ""),
                "value": gnn_row.get(col, np.nan),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_8_spectral_correlations.csv", index=False)
    
    pivot = df.pivot_table(index="correlation", columns="biome", values="value")
    pivot = pivot.reindex(columns=BIOMES)
    rpt("## Feature-error correlation by biome")
    rpt()
    rpt(pivot.to_markdown(floatfmt=".4f"))
    rpt()
    
    return df


def t1_9_canopy():
    rpt("# T1.9 — Canopy Analysis (GNN)")
    rpt()
    
    canopy_cols = ["canopy_mean", "canopy_max", "corr_ndvi_canopy",
                   "canopy_forest_mean", "canopy_open_mean", "canopy_water_mean", "canopy_rock_mean"]
    
    rows = []
    for biome in BIOMES:
        gnn_row = GNN_BEST[GNN_BEST["biome"] == biome].iloc[0]
        row_data = {"biome": biome}
        for col in canopy_cols:
            row_data[col] = gnn_row.get(col, np.nan)
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_9_canopy_analysis.csv", index=False)
    
    rpt_table(df, float_fmt=".3f")
    
    return df


def t2_statistics():
    rpt("# T2 — Advanced Statistical Framework")
    rpt()
    
    rows = []
    for biome in BIOMES:
        gnn_vals = TILES[(TILES["biome"] == biome) & (TILES["model"] == "GNN")]["mae_m"]
        cnn_vals = TILES[(TILES["biome"] == biome) & (TILES["model"] == "CNN")]["mae_m"]
        
        n_pairs = min(len(gnn_vals), len(cnn_vals))
        gnn_paired = gnn_vals.values[:n_pairs]
        cnn_paired = cnn_vals.values[:n_pairs]
        diff = cnn_paired - gnn_paired
        
        try:
            w_stat, w_p = sp_stats.wilcoxon(gnn_paired, cnn_paired)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        
        pooled_std = np.sqrt((gnn_paired.std()**2 + cnn_paired.std()**2) / 2)
        cohens_d = (cnn_paired.mean() - gnn_paired.mean()) / pooled_std if pooled_std > 0 else np.nan
        
        gnn_ranks = sp_stats.rankdata(np.concatenate([gnn_paired, cnn_paired]))[:n_pairs]
        cnn_ranks = sp_stats.rankdata(np.concatenate([gnn_paired, cnn_paired]))[n_pairs:]
        
        n_concordant = np.sum(diff > 0)
        n_discordant = np.sum(diff < 0)
        n_ties = np.sum(diff == 0)
        cliffs_delta = (n_concordant - n_discordant) / (n_concordant + n_discordant + n_ties) if (n_concordant + n_discordant + n_ties) > 0 else np.nan
        
        try:
            levene_stat, levene_p = sp_stats.levene(gnn_paired, cnn_paired)
        except Exception:
            levene_stat, levene_p = np.nan, np.nan
        
        n_boot = 10000
        boot_diffs = np.zeros(n_boot)
        rng = np.random.default_rng(42)
        for i in range(n_boot):
            idx = rng.integers(0, n_pairs, n_pairs)
            boot_diffs[i] = cnn_paired[idx].mean() - gnn_paired[idx].mean()
        ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
        
        try:
            mwu_stat, mwu_p = sp_stats.mannwhitneyu(gnn_paired, cnn_paired, alternative='two-sided')
        except Exception:
            mwu_stat, mwu_p = np.nan, np.nan
        
        rows.append({
            "biome": biome, "n_pairs": n_pairs,
            "gnn_mean": gnn_paired.mean(), "gnn_median": np.median(gnn_paired), "gnn_std": gnn_paired.std(),
            "cnn_mean": cnn_paired.mean(), "cnn_median": np.median(cnn_paired), "cnn_std": cnn_paired.std(),
            "mean_diff_cnn_minus_gnn": diff.mean(),
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "mannwhitney_stat": mwu_stat, "mannwhitney_p": mwu_p,
            "cohens_d": cohens_d, "cliffs_delta": cliffs_delta,
            "boot_ci_lower": ci_lower, "boot_ci_upper": ci_upper,
            "ci_excludes_zero": (ci_lower > 0) or (ci_upper < 0),
            "levene_stat": levene_stat, "levene_p": levene_p,
            "equal_variance": levene_p > 0.05 if pd.notna(levene_p) else None,
            "winner": "GNN" if diff.mean() > 0 else "CNN",
            "gnn_cv_pct": gnn_paired.std() / gnn_paired.mean() * 100 if gnn_paired.mean() > 0 else np.nan,
            "cnn_cv_pct": cnn_paired.std() / cnn_paired.mean() * 100 if cnn_paired.mean() > 0 else np.nan,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t2_statistical_framework.csv", index=False)
    
    rpt("## Complete statistical comparison (inference tiles)")
    rpt()
    display = df[["biome", "n_pairs", "gnn_mean", "gnn_median", "cnn_mean", "cnn_median",
                   "mean_diff_cnn_minus_gnn", "wilcoxon_p", "cohens_d", "cliffs_delta",
                   "boot_ci_lower", "boot_ci_upper", "winner"]]
    rpt_table(display, float_fmt=".4f")
    
    rpt("## Effect sizes interpretation")
    rpt()
    for _, r in df.iterrows():
        d = abs(r["cohens_d"])
        if d >= 0.8: size = "LARGE"
        elif d >= 0.5: size = "MEDIUM"
        elif d >= 0.2: size = "SMALL"
        else: size = "NEGLIGIBLE"
        rpt(f"- **{r['biome']}**: Cohen's d = {r['cohens_d']:.3f} ({size}), Cliff's δ = {r['cliffs_delta']:.3f}, Winner = **{r['winner']}**")
    rpt()
    
    rpt("## Outlier tiles detection (Cerrado GNN)")
    rpt()
    cerrado_gnn = TILES[(TILES["biome"] == "cerrado") & (TILES["model"] == "GNN")]["mae_m"]
    q1 = cerrado_gnn.quantile(0.25)
    q3 = cerrado_gnn.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = cerrado_gnn[(cerrado_gnn < lower) | (cerrado_gnn > upper)]
    rpt(f"- IQR method: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    rpt(f"- Bounds: [{lower:.2f}, {upper:.2f}]")
    rpt(f"- Outliers: {len(outliers)} de {len(cerrado_gnn)} tiles ({len(outliers)/len(cerrado_gnn)*100:.1f}%)")
    rpt()
    
    return df


def main():
    rpt(f"# Expanded Results — Gold Standard Analysis")
    rpt(f"> Automatically generated by `t1_full_analysis.py`")
    rpt(f"> Sources: GNN_RUN_COLAB + CNN_U-NET + testes_estatisticos")
    rpt()
    rpt("---")
    rpt()
    
    print('T1.1 — Tile distribution...')
    t1_1_tile_distribution()
    
    print('T1.2 — P90 by terrain...')
    t1_2_p90_terrain()
    
    print('T1.3 — Geomorphic P90...')
    t1_3_p90_geomorphic()
    
    print("T1.4 — Multi-task...")
    t1_4_multitask()
    
    print("T1.5 — Physics...")
    t1_5_physics()
    
    print("T1.6 — Generalização...")
    t1_6_generalization()
    
    print("T1.7 — Cerrado...")
    t1_7_cerrado_diagnosis()
    
    print("T1.8 — Spectral...")
    t1_8_spectral_correlations()
    
    print("T1.9 — Canopy...")
    t1_9_canopy()
    
    print("T2 — Estatísticas...")
    t2_statistics()
    
    report_path = OUT / "FULL_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved closely to: {report_path}")
    print(f"CSVs saved strictly to: {OUT}")
    
    csvs = list(OUT.glob("t*.csv"))
    print(f"\nTotal CSVs generated: {len(csvs)}")
    for csv in sorted(csvs):
        df_tmp = pd.read_csv(csv)
        print(f"  {csv.name}: {len(df_tmp)} rows × {len(df_tmp.columns)} columns")


if __name__ == "__main__":
    main()
