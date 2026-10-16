INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
from pathlib import Path
import platform
import os

if os.getenv('ARPIA_BASE_DIR'):
    BASE_DIR = Path(os.getenv('ARPIA_BASE_DIR'))
    if os.getenv('ARPIA_TILES_DIR'):
        TILES_DIR = Path(os.getenv('ARPIA_TILES_DIR'))
    else:
        TILES_DIR = BASE_DIR / "GRAPH_V18.3"
elif platform.system() == 'Windows':
    BASE_DIR = Path(f"{INTERNAL_DATA_ROOT}/v18.3")
    TILES_DIR = BASE_DIR / "GRAPH_V18.3" / "GINE_GRAPH"
else:
    BASE_DIR = Path(f"{INTERNAL_DATA_ROOT}")
    TILES_DIR = BASE_DIR / "processed_graph_v18.3"

DATASET_PATH = TILES_DIR / "mata_atlantica_v18.3_part2.pt"

BIOME_DATASETS = {
    'amazonia': TILES_DIR / "amazonia_v18.3_part2.pt",
    'cerrado': TILES_DIR / "cerrado_v18.3_part2.pt",
    'mata_atlantica': TILES_DIR / "mata_atlantica_v18.3_part2.pt",
    'pantanal': TILES_DIR / "pantanal_v18.3_part2.pt",
    'interior_sp': Path(f"{INTERNAL_DATA_ROOT}/processed_graph_v18.3/interior_sp_v18.3_part2.pt"),
}

REMAINING_BIOMES = ['cerrado', 'pantanal', 'interior_sp']

V21_BASE_DIR = Path(f"{INTERNAL_DATA_ROOT}/v21") if platform.system() == 'Windows' else Path(f"{INTERNAL_DATA_ROOT}/v21")
OUTPUT_DIR = V21_BASE_DIR / "GNN_RUN_COLAB" / "results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_CONFIG = {
    'batch_size': 1,
    'chunk_size': 350_000,
    
    'epochs_per_biome': 10,
    'biome': 'amazonia',
    
    'biome_order': ['amazonia', 'mata_atlantica', 'pantanal', 'cerrado'],
    'independent_biome_training': True,
    
    'num_workers': 8,
    'precision': '16-mixed',
    'pin_memory': True,
    'train_split': 0.8,
}

HETERO_CONFIG = {
    'node_types': ['dem', 'sentinel', 'lidar'],
    'metadata': {
        'dem': {'in_channels': 17},
        'sentinel': {'in_channels': 8},
        'lidar': {'in_channels': 3},
    },
    'edge_types': [
        ('dem', 'adjacent_to', 'dem'),
        ('sentinel', 'belongs_to', 'dem'),
        ('lidar', 'belongs_to', 'dem'),
        ('lidar', 'near_to', 'dem'),
        ('lidar', 'near_to', 'lidar')
    ]
}

NORMALIZATION_STATS = {
    'mata_atlantica': {
        'elev_mean': 363.58,  'elev_std': 76.22,
        'slope_max': 90.0,
        'is_normalized': True,
    },
    'amazonia': {
        'elev_mean': 78.25,  'elev_std': 35.28,
        'slope_max': 90.0,
        'is_normalized': True,
    },
    'cerrado': {
        'elev_mean': 350.64,  'elev_std': 76.40,
        'slope_max': 90.0,
        'is_normalized': True,
    },
    'pantanal': {
        'elev_mean': 106.71,  'elev_std': 28.74,
        'slope_max': 90.0,
        'is_normalized': True,
    },
    'interior_sp': {
        'elev_mean': 615.0,  'elev_std': 120.0,
        'slope_max': 90.0,
        'is_normalized': True,
    },
    'default': {
        'elev_mean': 500.0,  'elev_std': 300.0,
        'slope_max': 90.0,
        'is_normalized': False,
    }
}

MODEL_CONFIG = {
    'in_channels': 17,
    'hidden_channels': 512,
    'out_channels': 14,
    'num_layers': 6,
    'heads': 4,
    'dropout': 0.1,
    'edge_dim': 2,
}

BIOME_REGULARIZATION = {
    'cerrado': {
        'dropout': 0.35,
        'weight_decay': 2e-4,
        'label_smoothing': 0.15
    },
    'amazonia': {
        'dropout': 0.25,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1
    },
    'mata_atlantica': {
        'dropout': 0.20,
        'weight_decay': 5e-5,
        'label_smoothing': 0.08
    },
    'pantanal': {
        'dropout': 0.45,
        'weight_decay': 3e-4,
        'label_smoothing': 0.20
    }
}

OPTIMIZER_CONFIG = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'cyclical': True,
}

LOSS_CONFIG = {
    'elevation_weight': 1.0,
    'slope_weight': 0.5,
    'aspect_weight': 0.3,
    'curvature_weight': 0.3,
    'tpi_weight': 0.3,
    'tri_weight': 0.3,
    'roughness_weight': 0.3,
    'flow_accum_weight': 0.5,
    'consistency_weight': 0.1,
    'terrain_weight': 0.1,
    'sparse_weight': 20.0,
    'adaptive_sparse': True,
    'sparse_target_gap': 1.3,
    'sparse_min': 5.0,
    'sparse_max': 50.0,
    'steep_slope_weight': 1.0,
    'use_huber': True,
    'huber_delta': 1.0,
    'learnable_weights': True,
    'weight_min_clamp': 0.1,
    'weight_max_clamp': 2.0,
    
    'adaptive_spectral_gate': True,
    'spectral_gate_threshold': -0.3,
    'spectral_gate_penalty': 0.5,
    'spectral_gate_window': 5,
    
    'gap_early_stop': True,
    'gap_threshold': 3.0,
    'gap_patience': 5,
    
    'adaptive_gap_window': 5,
    
    'spectral_feature_indices': [8, 9, 10, 11, 12, 13],
    'spectral_base_weights': {
        8: 1.0,
        9: 1.0,
        10: 1.0,
        11: 1.0,
        12: 1.0,
        13: 1.0,
    },
}

TERRAIN_DERIVATIVE_CONFIG = {
    'slope': {
        'weight': 0.5,
        'warmup_epochs': 5,
        'active': True
    },
    'aspect': {
        'weight': 0.3,
        'warmup_epochs': 8,
        'active': True
    },
    'curvature': {
        'weight': 0.4,
        'warmup_epochs': 10,
        'active': True
    },
    'tpi': {
        'weight': 0.3,
        'warmup_epochs': 12,
        'active': True
    },
    'tri': {
        'weight': 0.3,
        'warmup_epochs': 12,
        'active': True
    },
    'roughness': {
        'weight': 0.2,
        'warmup_epochs': 15,
        'active': True
    },
    'flow_accum': {
        'weight': 0.2,
        'warmup_epochs': 18,
        'active': True
    }
}

def get_derivative_weight(epoch: int, derivative_name: str) -> float:
    config = TERRAIN_DERIVATIVE_CONFIG.get(derivative_name, {})
    if not config.get('active', False):
        return 0.0
    
    warmup = config.get('warmup_epochs', 0)
    weight = config.get('weight', 0.0)
    
    if epoch < warmup:
        return 0.0
    elif epoch < warmup + 5:
        progress = (epoch - warmup) / 5.0
        return weight * progress
    else:
        return weight

FEATURE_NAMES = [
    'elevation',
    'slope',
    'aspect_cos',
    'aspect_sin',
    'curvature',
    'tpi',
    'tri',
    'roughness',
    'b02_norm',
    'b03_norm',
    'b04_norm',
    'b08_norm',
    'ndvi',
    'ndwi',
    'has_lidar',
    'z_lidar',
    'confidence',
]

TARGET_NAMES = [
    'elevation',
    'slope',
    'aspect_cos',
    'aspect_sin',
    'curvature',
    'tpi',
    'tri',
    'roughness',
    'flow_accum',
    'canopy',
    'ndvi',
    'ndwi',
    'bsi',
    'shadow',
]

TARGET_INDICES = {
    'elevation': 0, 'slope': 1, 'aspect_cos': 2, 'aspect_sin': 3,
    'curvature': 4, 'tpi': 5, 'tri': 6, 'roughness': 7,
    'flow_accum': 8, 'canopy': 9,
    'ndvi': 10, 'ndwi': 11, 'bsi': 12, 'shadow': 13,
}

BIOME_LIMITS = {
    'amazonia':       (-50.0, 3600.0),
    'cerrado':        (0.0, 2400.0),
    'mata_atlantica': (-50.0, 3500.0),
    'pantanal':       (0.0, 500.0),
    'caatinga':       (0.0, 2500.0),
    'caatinga':       (0.0, 2500.0),
}

GLOBAL_LIMITS = (-100.0, 4000.0)

PHASE_CONFIGS = {
    
    '1_Elevation_Foundation': {
        'epochs': 20,
        'lr': 5e-5,
        'weights': {
            'elevation': 1.0,
            'slope': 0.5, 'aspect': 0.3, 'curvature': 0.3,
            'tpi': 0.3, 'tri': 0.3, 'roughness': 0.2, 'flow_accum': 0.2,
            'consistency': 0.05, 'terrain': 0.05,
            'spectral': 0.1
        }
    },
    
    '2_Slope_Physics': {
        'epochs': 20,
        'lr': 5e-5,
        'weights': {
            'elevation': 1.0,
            'slope': 1.5,
            'aspect': 0.3, 'curvature': 0.3,
            'tpi': 0.3, 'tri': 0.3, 'roughness': 0.2, 'flow_accum': 0.2,
            'consistency': 0.1, 'terrain': 0.1,
            'spectral': 0.2
        }
    },
    
    '3_Aspect_Orientation': {
        'epochs': 20,
        'lr': 5e-5,
        'weights': {
            'elevation': 1.0,
            'slope': 0.5,
            'aspect': 1.0,
            'curvature': 0.4,
            'tpi': 0.3, 'tri': 0.3, 'roughness': 0.2, 'flow_accum': 0.2,
            'consistency': 0.1, 'terrain': 0.1,
            'spectral': 0.3
        }
    },

    '4_Curvature_Topology': {
        'epochs': 25,
        'lr': 5e-5,
        'weights': {
            'elevation': 1.0,
            'slope': 1.0,
            'aspect': 1.0,
            'curvature': 1.0,
            'tpi': 0.5, 'tri': 0.5, 'roughness': 0.3, 'flow_accum': 0.3,
            'consistency': 0.15, 'terrain': 0.15,
            'spectral': 0.5
        }
    },

    '5_Morphology_Final': {
        'epochs': 25,
        'lr': 1e-5,
        'weights': {
            'elevation': 1.0,
            'slope': 0.8,
            'aspect': 0.8,
            'curvature': 0.5,
            'tpi': 0.5, 'tri': 0.5, 'roughness': 0.3, 'flow_accum': 0.5,
            'consistency': 0.15,
            'terrain': 0.15,
            'spectral': 0.5
        }
    },
}
