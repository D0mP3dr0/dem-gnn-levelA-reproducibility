import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

class HuberElevationLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.delta)

class LossTensorValidator:
    
    @staticmethod
    def validate_loss_tensors(predictions: torch.Tensor,
                             targets: torch.Tensor,
                             context: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        if predictions.dtype != torch.float32:
            predictions = predictions.float()
        if targets.dtype != torch.float32:
            targets = targets.float()
        
        if predictions.shape != targets.shape:
            if predictions.size(-1) > targets.size(-1):
                predictions = predictions[..., :targets.size(-1)]
            elif targets.size(-1) > predictions.size(-1):
                targets = targets[..., :predictions.size(-1)]
                
            if predictions.numel() == targets.numel():
                targets = targets.view(predictions.shape)

        predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return predictions, targets

class AdaptiveLossWeighting:
    
    def __init__(self, 
                 num_targets: int = 8,
                 target_names: Optional[List[str]] = None):
        self.num_targets = num_targets
        self.target_names = target_names or [
            'Elevation', 'Slope', 'Aspect', 'Curvature',
            'TPI', 'TRI', 'Roughness', 'Flow_Accum'
        ]
        
        self.weights = torch.ones(num_targets)
        self.critical_weights = {
            'Aspect': 2.0,
            'Flow_Accum': 2.0,
            'Slope': 1.5,
            'Curvature': 1.2,
        }
        
        for i, name in enumerate(self.target_names):
            if name in self.critical_weights:
                self.weights[i] = self.critical_weights[name]
    
    def get_weights(self, device: str = "cpu") -> torch.Tensor:
        return self.weights.to(device)

class TargetSpecificLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.huber = HuberElevationLoss(delta=1.0)
    
    def elevation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = LossTensorValidator.validate_loss_tensors(pred, target, "elevation")
        return self.huber(pred, target)
    
    def slope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = LossTensorValidator.validate_loss_tensors(pred, target, "slope")
        base_loss = F.smooth_l1_loss(pred, target, beta=1.0)
        
        if pred.size(0) > 1:
            pred_diff = torch.abs(pred[1:] - pred[:-1])
            target_diff = torch.abs(target[1:] - target[:-1])
            smoothness_loss = F.mse_loss(pred_diff, target_diff)
            return base_loss + 0.1 * smoothness_loss
        return base_loss
    
    def aspect_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = LossTensorValidator.validate_loss_tensors(pred, target, "aspect")
        
        if pred.max() > 2 * np.pi:
            pred = torch.deg2rad(pred)
            target = torch.deg2rad(target)
            
        pred = pred % (2 * np.pi)
        target = target % (2 * np.pi)
        
        pred_sin, pred_cos = torch.sin(pred), torch.cos(pred)
        target_sin, target_cos = torch.sin(target), torch.cos(target)
        
        loss = F.mse_loss(pred_sin, target_sin) + F.mse_loss(pred_cos, target_cos)
        return loss

    def flow_accum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = LossTensorValidator.validate_loss_tensors(pred, target, "flow_accum")
        return F.smooth_l1_loss(pred, target, beta=1.0)

    def standard_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = LossTensorValidator.validate_loss_tensors(pred, target, "standard")
        return F.mse_loss(pred, target)

class OptimizedARPIALoss(nn.Module):
    
    def __init__(self,
                 target_names: Optional[List[str]] = None,
                 adaptive_weighting: bool = True):
        super().__init__()
        
        self.target_names = target_names or [
            'Elevation', 'Slope', 'Aspect', 'Curvature',
            'TPI', 'TRI', 'Roughness', 'Flow_Accum'
        ]
        self.num_targets = len(self.target_names)
        self.adaptive_weighting = adaptive_weighting
        self.target_loss = TargetSpecificLoss()
        
        if self.adaptive_weighting:
            self.adaptive_weights = AdaptiveLossWeighting(
                num_targets=self.num_targets,
                target_names=self.target_names
            )
    
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                batch: Any) -> torch.Tensor:
        
        if isinstance(outputs, dict) and 'predictions' in outputs:
            predictions = outputs['predictions']
        else:
            predictions = outputs
            
        if hasattr(batch, 'y'):
            targets = batch.y
        else:
            targets = torch.zeros_like(predictions)

        if predictions.dim() != targets.dim():
             if predictions.dim() == 1: predictions = predictions.unsqueeze(-1)
             if targets.dim() == 1: targets = targets.unsqueeze(-1)
        
        device = predictions.device
        
        if predictions.size(1) > 1:
            losses = []
            for i in range(min(predictions.size(1), targets.size(1))):
                pred_i = predictions[:, i]
                target_i = targets[:, i]
                name = self.target_names[i] if i < len(self.target_names) else 'standard'
                
                if name == 'Elevation': val = self.target_loss.elevation_loss(pred_i, target_i)
                elif name == 'Slope': val = self.target_loss.slope_loss(pred_i, target_i)
                elif name == 'Aspect': val = self.target_loss.aspect_loss(pred_i, target_i)
                elif name == 'Flow_Accum': val = self.target_loss.flow_accum_loss(pred_i, target_i)
                else: val = self.target_loss.standard_loss(pred_i, target_i)
                
                losses.append(val)
            
            losses = torch.stack(losses)
            
            if self.adaptive_weighting:
                w = self.adaptive_weights.get_weights(device)
                w = w[:len(losses)]
                return (losses * w).sum() / w.sum()
            return losses.mean()
            
        else:
            return self.target_loss.elevation_loss(predictions, targets)

ARPIALoss = OptimizedARPIALoss
