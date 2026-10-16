import torch
import torch.nn as nn
from typing import Dict, Any

from .consistency_losses import MultiTaskLoss

class MultiTaskLossHetero(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_module = MultiTaskLoss(**kwargs)
        
    def forward(self, 
                preds: Dict[str, torch.Tensor], 
                batch: Any,
                epoch: int = None,
                warmup_epochs: int = 5) -> Dict[str, torch.Tensor]:
        
        dem_y = batch['dem'].y
        num_targets = dem_y.shape[1] if len(dem_y.shape) > 1 else 1
        
        targets = {
            'elevation': dem_y[:, 0].unsqueeze(1),
            'slope': dem_y[:, 1].unsqueeze(1),
        }
        
        if num_targets >= 4:
            targets['aspect_cos'] = dem_y[:, 2].unsqueeze(1)
            targets['aspect_sin'] = dem_y[:, 3].unsqueeze(1)
        
        if num_targets >= 8:
            targets['curvature'] = dem_y[:, 4].unsqueeze(1)
            targets['tpi'] = dem_y[:, 5].unsqueeze(1)
            targets['tri'] = dem_y[:, 6].unsqueeze(1)
            targets['roughness'] = dem_y[:, 7].unsqueeze(1)
        
        if num_targets >= 9:
            targets['flow_accum'] = dem_y[:, 8].unsqueeze(1)
        
        if num_targets >= 10:
            targets['canopy'] = dem_y[:, 9].unsqueeze(1)
        
        if num_targets >= 14:
            targets['ndvi'] = dem_y[:, 10].unsqueeze(1)
            targets['ndwi'] = dem_y[:, 11].unsqueeze(1)
            targets['bsi'] = dem_y[:, 12].unsqueeze(1)
            targets['shadow'] = dem_y[:, 13].unsqueeze(1)
        
        edge_index = batch['dem', 'adjacent_to', 'dem'].edge_index
        edge_attr = getattr(batch['dem', 'adjacent_to', 'dem'], 'edge_attr', None)
        
        class PseudoData:
            def __init__(self, x, edge_index, edge_attr):
                self.x = x
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                
        pseudo_data = PseudoData(batch['dem'].x, edge_index, edge_attr)
        
        
        return self.loss_module(preds, targets, pseudo_data, 
                              epoch=epoch, warmup_epochs=warmup_epochs)
