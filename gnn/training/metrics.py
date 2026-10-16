import torch
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)

class TensorValidator:
    
    @staticmethod
    def validate_and_prepare(predictions: torch.Tensor, 
                           targets: torch.Tensor,
                           context: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if predictions is None or targets is None:
                raise ValueError(f"Tensors cannot be None in {context}")
            
            if predictions.is_cuda:
                predictions = predictions.cpu()
            if targets.is_cuda:
                targets = targets.cpu()
            
            if predictions.dtype != torch.float32:
                predictions = predictions.float()
            if targets.dtype != torch.float32:
                targets = targets.float()
            
            if predictions.shape != targets.shape:
                if predictions.dim() == targets.dim():
                    min_size = min(predictions.size(0), targets.size(0))
                    predictions = predictions[:min_size]
                    targets = targets[:min_size]
                    
                    if predictions.dim() > 1 and predictions.shape != targets.shape:
                        min_cols = min(predictions.size(-1), targets.size(-1))
                        predictions = predictions[..., :min_cols]
                        targets = targets[..., :min_cols]
                else:
                    if predictions.numel() == targets.numel():
                        targets = targets.view(predictions.shape)
                    else:
                        min_elements = min(predictions.numel(), targets.numel())
                        predictions = predictions.flatten()[:min_elements]
                        targets = targets.flatten()[:min_elements]
            
            pred_invalid = torch.isnan(predictions) | torch.isinf(predictions)
            target_invalid = torch.isnan(targets) | torch.isinf(targets)
            invalid_mask = pred_invalid | target_invalid
            
            if invalid_mask.any():
                valid_mask = ~invalid_mask
                if valid_mask.sum() == 0:
                    raise ValueError(f"All values are invalid in {context}")
                
                predictions = predictions[valid_mask]
                targets = targets[valid_mask]
            
            if predictions.numel() == 0:
                raise ValueError(f"No valid data left in {context}")
            
            return predictions, targets
            
        except Exception as e:
            logger.error(f"Error validating tensors in {context}: {e}")
            raise

class TerrainMetrics:
    
    @staticmethod
    def elevation_accuracy(predictions: torch.Tensor, 
                          targets: torch.Tensor,
                          tolerance_meters: float = 1.0) -> Dict[str, float]:
        try:
            pred, target = TensorValidator.validate_and_prepare(predictions, targets, "elevation_accuracy")
            
            abs_error = torch.abs(pred - target)
            
            metrics = {
                'mae_meters': abs_error.mean().item(),
                'rmse_meters': torch.sqrt(torch.mean((pred - target) ** 2)).item(),
                'max_error_meters': abs_error.max().item(),
                'accuracy_within_tolerance': (abs_error <= tolerance_meters).float().mean().item(),
                'percentile_90_error': ARPIAMetrics._safe_quantile_static(abs_error, 0.9),
                'percentile_95_error': ARPIAMetrics._safe_quantile_static(abs_error, 0.95),
                'range_prediction': (pred.max() - pred.min()).item(),
                'range_target': (target.max() - target.min()).item()
            }
            return metrics
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def slope_accuracy(predictions: torch.Tensor, 
                      targets: torch.Tensor,
                      tolerance_degrees: float = 2.0) -> Dict[str, float]:
        try:
            pred, target = TensorValidator.validate_and_prepare(predictions, targets, "slope_accuracy")
            
            if pred.max() <= math.pi/2:
                pred_degrees = pred * 180 / math.pi
                target_degrees = target * 180 / math.pi
            else:
                pred_degrees = pred
                target_degrees = target
            
            abs_error = torch.abs(pred_degrees - target_degrees)
            
            metrics = {
                'mae_degrees': abs_error.mean().item(),
                'rmse_degrees': torch.sqrt(torch.mean((pred_degrees - target_degrees) ** 2)).item(),
                'max_error_degrees': abs_error.max().item(),
                'accuracy_within_tolerance': (abs_error <= tolerance_degrees).float().mean().item(),
                'slope_classification_accuracy': TerrainMetrics._slope_classification_accuracy(pred_degrees, target_degrees)
            }
            return metrics
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def aspect_accuracy(predictions: torch.Tensor, 
                       targets: torch.Tensor) -> Dict[str, float]:
        try:
            pred, target = TensorValidator.validate_and_prepare(predictions, targets, "aspect_accuracy")
            
            if pred.max() <= 2 * math.pi:
                pred_degrees = pred * 180 / math.pi
                target_degrees = target * 180 / math.pi
            else:
                pred_degrees = pred
                target_degrees = target
            
            diff = torch.abs(pred_degrees - target_degrees)
            circular_diff = torch.min(diff, 360 - diff)
            
            metrics = {
                'mae_degrees': circular_diff.mean().item(),
                'rmse_degrees': torch.sqrt(torch.mean(circular_diff ** 2)).item(),
                'accuracy_within_45deg': (circular_diff <= 45).float().mean().item(),
                'accuracy_within_90deg': (circular_diff <= 90).float().mean().item(),
                'directional_accuracy': TerrainMetrics._directional_accuracy(pred_degrees, target_degrees)
            }
            return metrics
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def _slope_classification_accuracy(pred_degrees: torch.Tensor, 
                                     target_degrees: torch.Tensor) -> float:
        try:
            def classify_slope(slopes):
                categories = torch.zeros_like(slopes, dtype=torch.long)
                categories[(slopes >= 0) & (slopes < 5)] = 0
                categories[(slopes >= 5) & (slopes < 15)] = 1
                categories[(slopes >= 15) & (slopes < 30)] = 2
                categories[(slopes >= 30) & (slopes < 45)] = 3
                categories[slopes >= 45] = 4
                return categories
            
            pred_classes = classify_slope(pred_degrees)
            target_classes = classify_slope(target_degrees)
            
            accuracy = (pred_classes == target_classes).float().mean().item()
            return accuracy
        except Exception:
            return 0.0
    
    @staticmethod
    def _directional_accuracy(pred_degrees: torch.Tensor, 
                            target_degrees: torch.Tensor) -> float:
        try:
            def get_quadrant(angles):
                angles = angles % 360
                quadrants = torch.zeros_like(angles, dtype=torch.long)
                quadrants[(angles >= 0) & (angles < 90)] = 0
                quadrants[(angles >= 90) & (angles < 180)] = 1
                quadrants[(angles >= 180) & (angles < 270)] = 2
                quadrants[(angles >= 270) & (angles < 360)] = 3
                return quadrants
            
            pred_quadrants = get_quadrant(pred_degrees)
            target_quadrants = get_quadrant(target_degrees)
            
            return accuracy
        except Exception:
            return 0.0

class HydrologyMetrics:
    
    @staticmethod
    def sinkhole_density(elevation: torch.Tensor) -> float:
        try:
            if elevation.dim() < 2: return 0.0
            
            is_batch = False
            if elevation.dim() == 4:
                x = elevation
            elif elevation.dim() == 3:
                x = elevation.unsqueeze(1)
            else:
                return 0.0
                
            neg_elev = -x
            max_pooled = F.max_pool2d(neg_elev, kernel_size=3, stride=1, padding=1)
            
            is_sinkhole = (neg_elev == max_pooled)
            
            return is_sinkhole.float().mean().item()
        except:
            return 0.0

    @staticmethod
    def roughness_spectrum(elevation: torch.Tensor) -> Dict[str, float]:
        try:
            if elevation.dim() < 2: return {}
            
            fft = torch.fft.fft2(elevation)
            magnitude = torch.abs(fft)
            
            total_energy = magnitude.sum()
            if total_energy == 0: return {'spectral_entropy': 0.0}
            
            prob = magnitude / total_energy
            entropy = -(prob * torch.log(prob + 1e-10)).sum()
            
            return {'spectral_entropy': entropy.item()} 
        except:
            return {}

class ARPIAMetrics:
    
    def __init__(self, target_names: Optional[List[str]] = None):
        self.target_names = target_names or [
            'Elevation', 'Slope', 'Aspect', 'Curvature', 
            'TPI', 'TRI', 'Roughness', 'Flow_Accum'
        ]
        self.elevation_tolerance = 1.0
        self.slope_tolerance = 2.0
        self.max_quantile_samples = 100000
    
    @staticmethod
    def _safe_quantile_static(tensor: torch.Tensor, q: float) -> float:
        try:
            max_samples = 100000
            if tensor.numel() > max_samples:
                indices = torch.randperm(tensor.numel())[:max_samples]
                sampled_tensor = tensor.flatten()[indices]
                return torch.quantile(sampled_tensor, q).item()
            else:
                return torch.quantile(tensor, q).item()
        except Exception:
            sorted_tensor, _ = torch.sort(tensor.flatten())
            idx = int(q * sorted_tensor.size(0))
            idx = max(0, min(idx, sorted_tensor.size(0) - 1))
            return sorted_tensor[idx].item()
    
    def compute_metrics(self, 
                       predictions: torch.Tensor, 
                       targets: torch.Tensor,
                       return_per_target: bool = True) -> Dict[str, Any]:
        metrics_result = {
            'overall': {},
            'per_target': {},
            'validation_info': {}
        }
        
        try:
            pred, target = TensorValidator.validate_and_prepare(predictions, targets, "compute_metrics")
            
            metrics_result['validation_info'] = {
                'samples_used': pred.size(0),
                'targets_count': pred.size(1) if pred.dim() > 1 else 1
            }
            
            metrics_result['overall'] = self._compute_overall_metrics(pred, target)
            
            if return_per_target and pred.dim() > 1:
                metrics_result['per_target'] = self._compute_per_target_metrics(pred, target)
            
            return metrics_result
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            metrics_result['error'] = str(e)
            return metrics_result
    
    def _compute_overall_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        try:
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            mse = F.mse_loss(pred_flat, target_flat).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            mae = F.l1_loss(pred_flat, target_flat).item()
            
            try:
                correlation_matrix = torch.corrcoef(torch.stack([pred_flat, target_flat]))
                correlation = correlation_matrix[0, 1]
                r2 = (correlation ** 2).item() if not torch.isnan(correlation) else 0.0
            except:
                r2 = 0.0
            
            abs_error = torch.abs(pred_flat - target_flat)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'max_error': abs_error.max().item(),
                'median_error': abs_error.median().item()
            }
        except Exception as e:
            return {'error': str(e)}

    def _compute_per_target_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Dict]:
        per_target_metrics = {}
        num_targets = min(predictions.size(1), len(self.target_names))
        
        for i in range(num_targets):
            target_name = self.target_names[i]
            try:
                pred_i = predictions[:, i]
                target_i = targets[:, i]
                
                basic = self._compute_basic_metrics(pred_i, target_i)
                specialized = {}
                
                if target_name == 'Elevation':
                    specialized = TerrainMetrics.elevation_accuracy(pred_i, target_i, self.elevation_tolerance)
                elif target_name == 'Slope':
                    specialized = TerrainMetrics.slope_accuracy(pred_i, target_i, self.slope_tolerance)
                elif target_name == 'Aspect':
                    specialized = TerrainMetrics.aspect_accuracy(pred_i, target_i)
                
                per_target_metrics[target_name] = {**basic, **specialized}
            except Exception as e:
                per_target_metrics[target_name] = {'error': str(e)}
        
        return per_target_metrics

    def _compute_basic_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        mse = F.mse_loss(pred, target).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        mae = F.l1_loss(pred, target).item()
        return {'rmse': rmse, 'mae': mae, 'mse': mse}


