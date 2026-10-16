import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
def main():
    configs = [
        "Full TerrainGNN",
        "No Sentinel-2",
        "No Water Flatten",
        "2 GAT layers",
        "Elevation only",
        "No Curriculum",
        "No Physics Loss"
    ]
    maes = [2.72, 2.68, 3.07, 3.44, 3.75, 3.76, 3.77]
    colors = ['#2ca02c', '#7f7f7f', '#fd8d3c', '#fd8d3c', '#e6550d', '#e6550d', '#a63603']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(configs))
    bars = ax.bar(x, maes, color=colors, width=0.6)
    baseline = 2.72
    ax.axhline(y=baseline, color='#f28e1c', linestyle='--', alpha=0.8, zorder=0)
    for bar, mae in zip(bars, maes):
        color_txt = "black"
        yval = bar.get_height()
        ax.annotate(f"{mae:.2f} m",
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color_txt)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("Elevation MAE (m)", fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(maes) * 1.15)
    ax.set_title("Ablation Study: Elevation Error (Amazon)", fontsize=12, pad=25)
    plt.subplots_adjust(top=0.82)
    article_fig = Path(r"./")
    article_fig.mkdir(parents=True, exist_ok=True)
    out_path = article_fig / "ablation_study_chart_300dpi.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_path}")
if __name__ == "__main__":
    main()
