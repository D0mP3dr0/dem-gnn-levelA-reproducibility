import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor

from .metrics import ARPIAMetrics
from .loss import OptimizedARPIALoss

logger = logging.getLogger(__name__)

class ExtendedMetricsCollector:
    
    def __init__(self):
        self.gradient_history = []
        self.weight_history = []
        self.lr_history = []
        self.batch_losses = []
        
    def collect_all_metrics(self, model: nn.Module, optimizer: optim.Optimizer, 
                          current_loss: float, epoch: int, batch_losses: List[float] = None) -> Dict[str, Any]:
        metrics = {}
        
        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        if grad_norms:
            metrics['gradient_norm_mean'] = np.mean(grad_norms)
            metrics['gradient_norm_max'] = np.max(grad_norms)
            
        metrics['current_loss'] = current_loss
        if batch_losses:
            metrics['batch_loss_std'] = np.std(batch_losses)
            
        if optimizer.param_groups:
             metrics['current_lr'] = optimizer.param_groups[0]['lr']
             
        return metrics

class BackgroundAnalyticsProcessor:
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    def submit_phase_analysis(self, phase_data: Dict):
        self.executor.submit(self._process, phase_data)
        
    def _process(self, data):
        pass

PHASE_SPECIFIC_CONFIGS = {
    '1_Elevation_Foundation': {
        'initial_lr': 0.00005,
        'loss_config': {
            'elevation_weight': 1.0,
            'slope_weight': 0.0,
            'aspect_weight': 0.0,
            'consistency_weight': 0.0,
        }
    },
    
    '2_Slope_Physics': {
        'initial_lr': 0.00005,
        'loss_config': {
            'elevation_weight': 0.2,
            'slope_weight': 1.5,
            'aspect_weight': 0.1,
            'consistency_weight': 0.5,
        }
    },
    
    '3_Aspect_Orientation': {
        'initial_lr': 0.00005,
        'loss_config': {
            'elevation_weight': 0.5,
            'slope_weight': 0.5,
            'aspect_weight': 1.0,
            'consistency_weight': 0.2,
        }
    },

    '4_Curvature_Topology': {
        'initial_lr': 0.00005,
        'loss_config': {
            'elevation_weight': 0.5,
            'slope_weight': 1.0,
            'aspect_weight': 1.0,
            'consistency_weight': 0.5,
            'terrain_weight': 1.0,
        }
    },

    '5_Morphology_TPI_TRI': {
        'initial_lr': 0.00001,
        'use_contour_edges': True,
        'loss_config': {
            'elevation_weight': 0.8,
            'slope_weight': 0.8,
            'aspect_weight': 0.8,
            'consistency_weight': 0.1,
            'terrain_weight': 0.5,
        }
    },
}
