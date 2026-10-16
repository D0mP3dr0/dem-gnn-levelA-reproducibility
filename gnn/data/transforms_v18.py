import torch
import logging
from torch_geometric.transforms import BaseTransform
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class HeteroElevationNorm(BaseTransform):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        
    def __call__(self, data):
        if 'dem' in data.node_types and data['dem'].num_nodes > 0:
            dem_elev = data['dem'].x[:, 0]
            data['dem'].x[:, 0] = (dem_elev - self.mean) / self.std
            
        if 'lidar' in data.node_types and data['lidar'].num_nodes > 0:
            lidar_z = data['lidar'].x[:, 0]
            data['lidar'].x[:, 0] = (lidar_z - self.mean) / self.std
            
        return data
        
    def forward(self, data):
        return self(data)

class HeteroSlopeNorm(BaseTransform):
    def __call__(self, data):
        if 'dem' in data.node_types and data['dem'].num_nodes > 0:
            slope = data['dem'].x[:, 1]
            data['dem'].x[:, 1] = slope / 90.0
        return data

    def forward(self, data):
        return self(data)

class HeteroTargetNorm(BaseTransform):
    def __init__(self, elev_mean: float, elev_std: float):
        self.elev_mean = elev_mean
        self.elev_std = elev_std
        
    def __call__(self, data):
        if 'dem' in data.node_types and hasattr(data['dem'], 'y') and data['dem'].y is not None:
             if data['dem'].y.size(1) > 0:
                 data['dem'].y[:, 0] = (data['dem'].y[:, 0] - self.elev_mean) / self.elev_std
             
             if data['dem'].y.size(1) > 1:
                 data['dem'].y[:, 1] = data['dem'].y[:, 1] / 90.0
                 
        return data

    def forward(self, data):
        return self(data)

class HeteroComposer:
    @staticmethod
    def get_transforms(stats: Dict):
        from torch_geometric.transforms import Compose, ToDevice
        
        mean = stats.get('elev_mean', 0.0)
        std = stats.get('elev_std', 1.0)
        
        return Compose([
            HeteroElevationNorm(mean, std),
            HeteroSlopeNorm(),
            HeteroTargetNorm(mean, std)
        ])
