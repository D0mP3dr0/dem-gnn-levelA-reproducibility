INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
INTERNAL_GRAPH_ROOT = __import__("os").environ.get("DEMGNN_GRAPH_ROOT", ".")
import os

DATA_DIR = rf"{INTERNAL_GRAPH_ROOT}\GINE_GRAPH"
OUT_DIR = rf"{INTERNAL_DATA_ROOT}\kriging\results"

BIOMES = [
    "amazonia",
    "cerrado",
    "mata_atlantica",
    "pantanal"
]

TEST_PART = "part4"

N_LIDAR_SAMPLES = 5000
TILE_GRID_H = 10
TILE_GRID_W = 12
