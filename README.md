# dem-gnn-levelA-reproducibility

Code of the study *Graph Attention Networks for Multi-Task Terrain Reconstruction from Sparse Spaceborne LiDAR*
(L. F. C. Seelig and R. M. Salles).

The study reconstructs a terrain surface and its derivatives (slope, aspect, curvature, TPI, TRI, roughness,
flow accumulation) from sparse ICESat-2 and GEDI footprints with a heterogeneous graph attention network (GATv2)
over DEM, Sentinel-2 and LiDAR nodes, trained with a five-phase curriculum and physics-consistency losses, in four
Brazilian biomes (Amazonia, Cerrado, Mata Atlantica, Pantanal). Ordinary Kriging, a Random Forest and a residual
U-Net are the baselines.

This repository holds the scripts as they were run and the result records of the study. `PROVENANCE.md` gives, for
every file, its source, the evidence that it is the version that ran, and the SHA-256 of the file as run and as
published. Thirty of the thirty-five code files are traced to the version that ran; the other five are published
as released earlier and are marked *origin not identified*.

## Layout

```
gnn/        graph network (entry point gnn/main_v18.3.py; configuration gnn/config.py)
cnn/        residual U-Net baseline (entry point cnn/train.py; cnn/run_all_biomes.bat)
baselines/  Kriging, Random Forest and fair-evaluation scripts
analysis/   extraction of best-epoch metrics, statistical analysis and figures
hparams/    hyperparameter files written by the training runs
results/    training logs, per-tile baseline metrics and statistical tests (CSV)
```

The scripts read their data roots from two environment variables: `DEMGNN_DATA_ROOT` (working folder of the
pipeline) and `DEMGNN_GRAPH_ROOT` (folder of the preprocessed graphs). Both default to the current folder.

## Selected results

Best-epoch elevation error of the graph network on the graph validation nodes, in metres
(`results/statistical_tests/gnn_best_epoch_all_metrics.csv`):

| Biome | MAE | RMSE |
| --- | --- | --- |
| Amazonia | 2.72 | 3.49 |
| Cerrado | 2.21 | 2.77 |
| Mata Atlantica | 2.10 | 3.03 |
| Pantanal | 2.81 | 9.84 |

Paired comparison of tile-level MAE (m) against the U-Net on the inference tiles, with the Wilcoxon signed-rank
test (`results/statistical_tests/t2_statistical_framework.csv`):

| Biome | Tiles | GNN mean MAE | CNN mean MAE | Wilcoxon p | Lower error |
| --- | --- | --- | --- | --- | --- |
| Amazonia | 120 | 2.52 | 9.62 | 2.0e-21 | GNN |
| Cerrado | 120 | 13.41 | 2.86 | 2.2e-21 | CNN |
| Mata Atlantica | 118 | 3.24 | 16.17 | 9.8e-21 | GNN |
| Pantanal | 120 | 0.63 | 3.19 | 5.3e-19 | GNN |

## Input data

All inputs are public: SRTM v3 (USGS Earth Explorer, https://earthexplorer.usgs.gov), ICESat-2 ATL08 (NSIDC,
https://nsidc.org), GEDI L2A (NASA LP DAAC, https://lpdaac.usgs.gov), Sentinel-2 L2A and Copernicus DEM GLO-30
(Copernicus Data Space Ecosystem, https://dataspace.copernicus.eu).

The derived data (preprocessed heterogeneous graphs, fused imagery and trained checkpoints) exceed 60 GB and are
not in this repository; they are available on request from the authors.

## Verification

```bash
sha256sum -c SHA256SUMS        # every file, byte for byte
python3.11 verify_code.py      # executable code identical to the code as run (CODE_AST_SHA256)
python3 verify_payload.py      # records and the numbers above identical to the records as written (PAYLOAD_SHA256)
```

`verify_code.py` and `verify_payload.py` document what was normalized before publication (comments and
docstrings removed, messages translated into English, absolute data roots replaced by the variables above).

## Requirements

```bash
pip install -r requirements.txt
```

## License

Code: MIT (`LICENSE`). Result records under `results/` and `hparams/`: CC BY 4.0 (`LICENSE-DATA`).
