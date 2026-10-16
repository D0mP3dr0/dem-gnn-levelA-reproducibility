from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "fig"
RESULTS = ROOT / "resultados_expandidos"
DATA_REV = ROOT / "internal" / "statistical_tests"
BASELINES = ROOT / "internal" / "baselines"
BIOME_ORDER = ["amazonia", "cerrado", "mata_atlantica", "pantanal"]
BIOME_LABEL = {
    "amazonia": "Amazon",
    "cerrado": "Cerrado",
    "mata_atlantica": "Atlantic Forest",
    "pantanal": "Pantanal",
}
DPI = 300
mpl.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    }
)
def _ensure_out():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
def plot_multitask_heatmap():
    df = pd.read_csv(RESULTS / "t1_4_multitask_performance.csv")
    df = df[df["task"] != "aspect_deg"]
    tasks = [
        "elevation",
        "slope",
        "curvature",
        "tpi",
        "tri",
        "roughness",
        "flow_accum",
    ]
    mat = []
    for t in tasks:
        row = []
        for b in BIOME_ORDER:
            v = df[(df["biome"] == b) & (df["task"] == t)]["mae"].values
            row.append(float(v[0]) if len(v) else np.nan)
        mat.append(row)
    Z = np.array(mat, dtype=float)
    Z_plot = np.log10(Z + 1e-3)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(Z_plot, aspect="auto", cmap="viridis", origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(\mathrm{MAE} + 10^{-3})$")
    ax.set_xticks(range(len(BIOME_ORDER)))
    ax.set_xticklabels([BIOME_LABEL[b] for b in BIOME_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.replace("_", " ") for t in tasks])
    ax.set_title("GNN multi-task validation MAE (best epoch by biome)")
    task_units = {
        "elevation": "m",
        "slope": "°",
        "curvature": "—",
        "tpi": "m",
        "tri": "m",
        "roughness": "m",
        "flow_accum": "u",
    }
    for i in range(len(tasks)):
        for j in range(len(BIOME_ORDER)):
            u = task_units[tasks[i]]
            txt = f"{Z[i, j]:.3g}" if u != "°" else f"{Z[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=7, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Units: elevation, TPI, TRI, roughness in m; slope in degrees; flow_accum in model units; curvature dimensionless.",
        ha="center",
        fontsize=7,
        style="italic",
    )
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(OUT_FIG / "multitask_mae_heatmap_300dpi.png")
    plt.close('all')
def plot_gap_ratio_gnn():
    df = pd.read_csv(RESULTS / "gnn_best_epoch_all_metrics.csv")
    rows = []
    for b in BIOME_ORDER:
        r = df[df["biome"] == b].iloc[0]
        rows.append(
            {
                "biome": b,
                "mae_lidar": r["mae_lidar"],
                "mae_no_lidar": r["mae_no_lidar"],
                "gap_ratio": r["gap_ratio"],
            }
        )
    dfp = pd.DataFrame(rows)
    x = np.arange(len(BIOME_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - w / 2, dfp["mae_lidar"], w, label=r"MAE$_{\mathrm{LiDAR}}$", color="#2c7fb8")
    ax.bar(x + w / 2, dfp["mae_no_lidar"], w, label=r"MAE$_{\mathrm{no\,LiDAR}}$", color="#7fcdbb")
    ax.set_xticks(x)
    ax.set_xticklabels([BIOME_LABEL[b] for b in BIOME_ORDER])
    ax.set_ylabel("MAE (m)")
    ax.set_title(
        "GNN Generalization: MAE on nodes with vs. without LiDAR\n(validation, best epoch)",
        fontsize=9,
        pad=40,
    )
    ax.legend(
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
    )
    tops = np.maximum(dfp["mae_lidar"].values, dfp["mae_no_lidar"].values)
    y_hi = float(np.max(tops))
    for i, r in dfp.iterrows():
        y_top = max(r["mae_lidar"], r["mae_no_lidar"])
        ax.annotate(
            f"ratio {r['gap_ratio']:.3f}",
            xy=(i, y_top),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#333333",
        )
    ax.set_ylim(0, y_hi * 1.35)
    fig.text(
        0.5,
        0.02,
        r"Gap ratio = MAE$_{\mathrm{no\,LiDAR}}$ / MAE$_{\mathrm{LiDAR}}$; values < 1 indicate lower error off-transect.",
        ha="center",
        fontsize=7,
        style="italic",
    )
    plt.subplots_adjust(top=0.82, bottom=0.2)
    fig.savefig(OUT_FIG / "gnn_gap_ratio_anchors_300dpi.png", bbox_inches="tight", pad_inches=0.1)
    plt.close('all')
def plot_inference_tile_comparison():
    gnn_df = pd.read_csv(BASELINES / "gnn_official_fair_metrics.csv")
    model_data = {
        "GNN": {"color": "#88419d", "vals": []},
        "CNN (U-Net)": {"color": "#f28e1c", "vals": []},
        "Random Forest": {"color": "#2ca02c", "vals": []},
        "Kriging": {"color": "#1f77b4", "vals": []},
    }
    for b in BIOME_ORDER:
        row_gnn = gnn_df[gnn_df["biome"] == b]
        model_data["GNN"]["vals"].append(float(row_gnn["mae_m"].iloc[0]) if not row_gnn.empty else np.nan)
        cnn_path = BASELINES / "cnn" / f"cnn_{b}_sr_golden_metrics.csv"
        if cnn_path.exists():
            row_cnn = pd.read_csv(cnn_path)
            model_data["CNN (U-Net)"]["vals"].append(float(row_cnn["mae_m"].iloc[0]))
        else:
            model_data["CNN (U-Net)"]["vals"].append(np.nan)
        rf_path = BASELINES / "rf" / f"rf_{b}_golden_metrics.csv"
        if rf_path.exists():
            row_rf = pd.read_csv(rf_path)
            model_data["Random Forest"]["vals"].append(float(row_rf["mae_m"].iloc[0]))
        else:
            model_data["Random Forest"]["vals"].append(np.nan)
        krig_path = BASELINES / "kriging" / f"kriging_{b}_golden_metrics.csv"
        if krig_path.exists():
            row_k = pd.read_csv(krig_path)
            model_data["Kriging"]["vals"].append(float(row_k["mae_m"].iloc[0]))
        else:
            model_data["Kriging"]["vals"].append(np.nan)
    n_models = len(model_data)
    x = np.arange(len(BIOME_ORDER))
    w = 0.18
    offsets = np.linspace(-(n_models - 1) * w / 2, (n_models - 1) * w / 2, n_models)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i, (label, info) in enumerate(model_data.items()):
        bars = ax.bar(x + offsets[i], info["vals"], w, label=label, color=info["color"])
        for bar, val in zip(bars, info["vals"]):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color="#333333",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([BIOME_LABEL[b] for b in BIOME_ORDER])
    ax.set_ylabel("Elevation MAE (m)")
    ax.set_title(
        "Golden Sector Evaluation — MAE comparison (part2, highest LiDAR density)\n"
        "GNN vs CNN (U-Net) vs Random Forest vs Kriging",
        fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.text(
        0.5, 0.01,
        'Source: baselines/ — golden sector (corrected, no double-normalization).',
        ha="center", fontsize=6, style="italic",
    )
    plt.subplots_adjust(bottom=0.14)
    fig.savefig(OUT_FIG / "inference_tile_mae_gnn_vs_cnn_300dpi.png", bbox_inches="tight", pad_inches=0.1)
    plt.close('all')
def plot_spectral_gating():
    cols = ["gating_efficiency", "false_correction_rate", "spectral_bias_ratio"]
    df = pd.read_csv(RESULTS / "gnn_best_epoch_all_metrics.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(BIOME_ORDER))
    w = 0.25
    for k, c in enumerate(cols):
        vals = [float(df[df["biome"] == b][c].iloc[0]) for b in BIOME_ORDER]
        ax.bar(x + (k - 1) * w, vals, w, label=c.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([BIOME_LABEL[b] for b in BIOME_ORDER])
    ax.set_ylabel("Value")
    ax.set_title("Spectral gating metrics (GNN, best epoch)", pad=35)
    ax.legend(
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
    )
    y_max = max(float(df[df["biome"] == b][cols].iloc[0].max()) for b in BIOME_ORDER)
    ax.set_ylim(0, y_max * 1.12)
    plt.subplots_adjust(top=0.82)
    fig.savefig(OUT_FIG / "spectral_gating_by_biome_300dpi.png", bbox_inches="tight", pad_inches=0.1)
    plt.close('all')
def plot_p90_terrain_heatmap():
    df = pd.read_csv(RESULTS / "t1_2_p90_terrain_stratified.csv")
    terrains = sorted(df["terrain_category"].unique())
    biomes = BIOME_ORDER
    M = np.zeros((len(terrains), len(biomes)))
    for i, t in enumerate(terrains):
        for j, b in enumerate(biomes):
            sub = df[(df["terrain_category"] == t) & (df["biome"] == b)]
            if sub.empty:
                M[i, j] = np.nan
                continue
            g = sub["gnn_p90_m"].iloc[0]
            c = sub["cnn_p90_m"].iloc[0]
            if c > 1e-6:
                M[i, j] = 100.0 * (g - c) / c
            else:
                M[i, j] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    vmax = np.nanpercentile(np.abs(M), 90) if np.any(np.isfinite(M)) else 50
    vmax = max(vmax, 10)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"P90: $100 \times (\mathrm{GNN}-\mathrm{CNN})/\mathrm{CNN}$ (%)")
    ax.set_xticks(range(len(biomes)))
    ax.set_xticklabels([BIOME_LABEL[b] for b in biomes], rotation=20, ha="right")
    ax.set_yticks(range(len(terrains)))
    ax.set_yticklabels([t.replace("_", " ") for t in terrains])
    ax.set_title("P90 error by terrain category (validation/logs)")
    fig.text(
        0.5,
        0.02,
        "Blue: GNN better; Red: CNN better. Source: P90 metrics by terrain mask.",
        ha="center",
        fontsize=7,
        style="italic",
    )
    plt.subplots_adjust(bottom=0.2)
    fig.savefig(OUT_FIG / "p90_terrain_gnn_vs_cnn_heatmap_300dpi.png")
    plt.close('all')
def main():
    _ensure_out()
    print("Saída:", OUT_FIG)
    plot_multitask_heatmap()
    print("  OK multitask_mae_heatmap_300dpi.png")
    plot_gap_ratio_gnn()
    print("  OK gnn_gap_ratio_anchors_300dpi.png")
    plot_inference_tile_comparison()
    print('  OK inference_tile_mae_gnn_vs_cnn_300dpi.png  [4 models: GNN, CNN, RF, Kriging — corrected golden sector]')
    plot_spectral_gating()
    print("  OK spectral_gating_by_biome_300dpi.png")
    plot_p90_terrain_heatmap()
    print("  OK p90_terrain_gnn_vs_cnn_heatmap_300dpi.png")
    print("Concluído.")
if __name__ == "__main__":
    main()
