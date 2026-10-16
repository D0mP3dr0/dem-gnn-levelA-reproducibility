import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class HuberElevationLoss(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.dim() == 1 and targets.dim() == 2 and targets.size(1) > 0:
            targets = targets[:, 0]
        if targets.dim() == 1 and preds.dim() == 2 and preds.size(1) == 1:
            preds = preds.squeeze(1)
        diff = preds - targets
        abs_diff = diff.abs()
        quad = 0.5 * diff.pow(2)
        lin = self.delta * (abs_diff - 0.5 * self.delta)
        loss = torch.where(abs_diff <= self.delta, quad, lin)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class ConsistencyLoss(nn.Module):
    
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        
    def spatial_consistency_loss(self, elevation_pred: torch.Tensor,
                                edge_index: torch.Tensor,
                                edge_attr: Optional[torch.Tensor] = None,
                                weight: float = 1.0) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
        
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        
        src_elevations = elevation_pred[src_idx].squeeze()
        tgt_elevations = elevation_pred[tgt_idx].squeeze()
        
        pred_elevation_diff = torch.abs(src_elevations - tgt_elevations)
        
        if edge_attr is not None and edge_attr.size(1) > 0:
            distances = edge_attr[:, 0]
            max_distance = distances.max()
            if max_distance > 0:
                distance_weights = distances / max_distance
            else:
                distance_weights = torch.ones_like(distances)
            
            if edge_attr.size(1) > 3:
                terrain_difficulty = edge_attr[:, 3]
                difficulty_weights = 1.0 + terrain_difficulty
            else:
                difficulty_weights = torch.ones_like(distances)
            
            edge_weights = distance_weights * difficulty_weights
        else:
            edge_weights = torch.ones_like(pred_elevation_diff)
        
        consistency_loss = (edge_weights * pred_elevation_diff ** 2).mean()
        return weight * consistency_loss

    def water_flow_loss(self, elevation_pred: torch.Tensor,
                        edge_index: torch.Tensor,
                        water_mask: torch.Tensor,
                        weight: float = 1.0) -> torch.Tensor:
        if water_mask.sum() == 0 or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
            
        return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)

class SpectralGatingLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, canopy_pred: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if canopy_pred is None:
            return torch.tensor(0.0, device=x.device), {}
            
        ndvi = x[:, 12]
        
        mask_open = ndvi < 0.2
        
        mask_invalid = mask_open
        if x.shape[1] > 13:
            ndwi = x[:, 13]
            mask_water = ndwi > 0.1
            mask_invalid = mask_invalid | mask_water
            
        if mask_invalid.sum() == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True), {'spectral_gating_val': 0.0}
            
        invalid_corrections = canopy_pred[mask_invalid]
        loss = (invalid_corrections ** 2).mean()
        
        return self.weight * loss, {'spectral_gating_val': loss.item()}
        is_water = (water_mask > 0.5).squeeze()
        
        if is_water.sum() == 0 or edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
            
        src, dst = edge_index
        
        water_src_mask = is_water[src]
        
        if not water_src_mask.any():
             return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
             
        elev_water = elevation_pred[src][water_src_mask]
        elev_neighbor = elevation_pred[dst][water_src_mask]
        
        violation = F.relu(elev_water - elev_neighbor)
        
        return weight * violation.mean()
    
class TerrainLoss(nn.Module):
    
    def __init__(self):
        super(TerrainLoss, self).__init__()
        
    def terrain_consistency_loss(self, elevation_pred: torch.Tensor,
                                slope_pred: torch.Tensor,
                                aspect_pred: torch.Tensor,
                                edge_index: torch.Tensor,
                                coords: Optional[torch.Tensor] = None,
                                weight: float = 1.0,
                                elev_mean: float = 0.0,
                                elev_std: float = 1.0) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
        
        slope_loss = self._slope_consistency_loss(
            elevation_pred, slope_pred, edge_index, coords,
            elev_mean=elev_mean, elev_std=elev_std
        )
        total_loss = total_loss + slope_loss
        
        aspect_loss = self._aspect_consistency_loss(elevation_pred, aspect_pred, edge_index, coords)
        total_loss = total_loss + aspect_loss
        
        return weight * total_loss
    
    def _slope_consistency_loss(self, elevation_pred: torch.Tensor,
                               slope_pred: torch.Tensor,
                               edge_index: torch.Tensor,
                               coords: Optional[torch.Tensor] = None,
                               elev_mean: float = 0.0,
                               elev_std: float = 1.0) -> torch.Tensor:
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        
        src_elevations_real = elevation_pred[src_idx].squeeze() * elev_std + elev_mean
        tgt_elevations_real = elevation_pred[tgt_idx].squeeze() * elev_std + elev_mean
        elevation_diff = torch.abs(src_elevations_real - tgt_elevations_real)
        
        if coords is not None:
            src_coords = coords[src_idx]
            tgt_coords = coords[tgt_idx]
            distances = torch.norm(src_coords - tgt_coords, dim=1)
            gradient = elevation_diff / (distances + 1e-8)
            calculated_slopes = torch.atan(gradient) * (180.0 / 3.14159)
        else:
            calculated_slopes = elevation_diff
        
        src_slopes_deg = slope_pred[src_idx].squeeze() * 90.0
        slope_loss = F.mse_loss(src_slopes_deg, calculated_slopes)
        return slope_loss
    
    def _aspect_consistency_loss(self, elevation_pred: torch.Tensor,
                                aspect_pred: torch.Tensor,
                                edge_index: torch.Tensor,
                                coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords is None:
            return torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
        
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        
        src_coords = coords[src_idx]
        tgt_coords = coords[tgt_idx]
        
        direction = tgt_coords - src_coords
        
        calculated_aspect = torch.atan2(direction[:, 1], direction[:, 0])
        calculated_aspect = calculated_aspect * 180.0 / torch.pi
        calculated_aspect = (calculated_aspect + 360) % 360
        
        src_aspects = aspect_pred[src_idx].squeeze()
        
        aspect_diff = torch.abs(src_aspects - calculated_aspect)
        aspect_diff = torch.min(aspect_diff, 360 - aspect_diff)
        
        aspect_loss = aspect_diff.mean()
        return aspect_loss

class GeomorphLoss(nn.Module):
    def __init__(self, 
                 water_weight: float = 1.0, 
                 river_width_weight: float = 0.5,
                 cliff_weight: float = 1.0,
                 bio_weight: float = 0.5,
                 plateau_weight: float = 3.0,
                 valley_weight: float = 2.0):
        super().__init__()
        self.water_weight = water_weight
        self.river_width_weight = river_width_weight
        self.cliff_weight = cliff_weight
        self.bio_weight = bio_weight
        self.plateau_weight = plateau_weight
        self.valley_weight = valley_weight
        
    def forward(self, 
                elevation_pred: torch.Tensor,
                slope_pred: torch.Tensor,
                edge_index: torch.Tensor,
                data
                ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = torch.tensor(0.0, device=elevation_pred.device, requires_grad=True)
        
        if edge_index.shape[1] == 0:
            return total_loss, losses

        
        has_ndwi = data.x.size(1) > 13
        has_ndvi = data.x.size(1) > 12
        has_texture = data.x.size(1) > 7
        
        if has_ndwi:
            ndwi = data.x[:, 13]
            water_mask = (ndwi > 0.1).float()
            is_wet = water_mask
            is_dry = 1.0 - is_wet
        else:
            is_wet = torch.zeros_like(elevation_pred)
            is_dry = torch.ones_like(elevation_pred)
            
        
        if has_ndwi and is_wet.sum() > 0:
            src, dst = edge_index
            
            edge_is_wet_src = is_wet[src] > 0.5
            
            if edge_is_wet_src.any():
                elev_src = elevation_pred[src][edge_is_wet_src]
                elev_dst = elevation_pred[dst][edge_is_wet_src]
                
                flow_violation = F.relu(elev_src - elev_dst)
                loss_flow = flow_violation.mean()
                
                losses['water_flow'] = loss_flow
                losses['water_flow_edges'] = edge_is_wet_src.sum().item()
                total_loss = total_loss + self.water_weight * loss_flow


        
        if has_texture:
            roughness = data.x[:, 6]
            cliff_context = is_dry * (roughness > 0.5).float()
            
            if cliff_context.sum() > 0:
                cliff_flat_penalty = F.relu(5.0 - slope_pred.squeeze()) * cliff_context
                loss_cliff = cliff_flat_penalty.mean()
                
                losses['cliff_structure'] = loss_cliff
                losses['cliff_context_nodes'] = (cliff_context > 0.5).sum().item()
                total_loss = total_loss + self.cliff_weight * loss_cliff

        
        if has_ndvi and has_ndwi:
            ndvi = data.x[:, 12]
            plateau_context = is_dry * (ndvi < 0.2).float()
            
            if plateau_context.sum() > 0:
                plateau_slope_loss = (slope_pred.squeeze() * plateau_context).abs().mean()
                
                losses['dry_plateau'] = plateau_slope_loss
                losses['plateau_context_nodes'] = (plateau_context > 0.5).sum().item()
                total_loss = total_loss + self.plateau_weight * plateau_slope_loss
        
        
        has_curvature = data.x.size(1) > 4
        
        if has_curvature and edge_index.shape[1] > 0:
            curvature = data.x[:, 4]
            is_concave = (curvature < -0.005).float()
            valley_context = is_dry * is_concave
            
            if valley_context.sum() > 0:
                src, dst = edge_index
                elev_pred = elevation_pred.squeeze()
                num_nodes = elev_pred.size(0)
                
                min_neighbor = torch.full((num_nodes,), float('inf'), device=elev_pred.device)
                min_neighbor = min_neighbor.scatter_reduce(
                    0, src, elev_pred[dst].detach(), reduce='amin', include_self=False
                )
                
                valley_violation = F.relu(elev_pred - min_neighbor) * valley_context
                loss_valley = valley_violation.mean()
                
                losses['curvature_valley'] = loss_valley
                losses['valley_context_nodes'] = (valley_context > 0.5).sum().item()
                total_loss = total_loss + self.valley_weight * loss_valley
                
        losses['geomorph_total'] = total_loss
        return total_loss, losses

class SparseAnchorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, sparse_y: torch.Tensor, sparse_mask: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        if sparse_mask is None or not sparse_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        if pred.dim() > 1:
            pred_flat = pred.squeeze()
        else:
            pred_flat = pred
            
        anchor_preds = pred_flat[sparse_mask]
        anchor_targets = sparse_y[sparse_mask]
        
        loss = F.mse_loss(anchor_preds, anchor_targets)
        return weight * loss

class MultiTaskLoss(nn.Module):
    
    def __init__(self, 
                 elevation_weight: float = 1.0,
                 slope_weight: float = 0.3,
                 aspect_weight: float = 0.3,
                 consistency_weight: float = 0.1,
                 terrain_weight: float = 0.1,
                 sparse_weight: float = 50.0,
                 adaptive_weighting: bool = False,
                 target_names: Optional[List[str]] = None,
                 **kwargs):
        super(MultiTaskLoss, self).__init__()
        
        self.elevation_weight = elevation_weight
        self.slope_weight = slope_weight
        self.aspect_weight = aspect_weight
        self.consistency_weight = consistency_weight
        self.terrain_weight = terrain_weight
        self.sparse_weight = sparse_weight
        
        if adaptive_weighting:
            logger.info("MultiTaskLoss: 'adaptive_weighting' passed in config but using fixed weights for stability.")
        
        self.consistency_loss = ConsistencyLoss()
        self.terrain_loss = TerrainLoss()
        self.sparse_loss = SparseAnchorLoss()
        self.adaptive_loss_module = AdaptiveGeomorphLoss()
        self.geomorph_loss = GeomorphLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                data,
                coarse_anchor: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None,
                warmup_epochs: int = 5) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = torch.tensor(0.0, device=predictions['elevation'].device, requires_grad=True)
        
        if epoch is not None and epoch < warmup_epochs and self.consistency_weight > 0:
            ramp = (epoch + 1) / warmup_epochs
            effective_consistency_weight = self.consistency_weight * ramp
        else:
            effective_consistency_weight = self.consistency_weight
        
        elevation_loss = F.mse_loss(predictions['elevation'], targets['elevation'])
        losses['elevation'] = elevation_loss
        total_loss = total_loss + self.elevation_weight * elevation_loss
        
        if 'slope' in predictions and 'slope' in targets:
            slope_loss = F.mse_loss(predictions['slope'], targets['slope'])
            losses['slope'] = slope_loss
            total_loss = total_loss + self.slope_weight * slope_loss
        
        if 'aspect_cos' in predictions and 'aspect_cos' in targets:
            aspect_cos_loss = F.mse_loss(predictions['aspect_cos'], targets['aspect_cos'])
            aspect_sin_loss = F.mse_loss(predictions['aspect_sin'], targets['aspect_sin'])
            aspect_loss = (aspect_cos_loss + aspect_sin_loss) / 2.0
            losses['aspect'] = aspect_loss
            total_loss = total_loss + self.aspect_weight * aspect_loss
        elif 'aspect' in predictions and 'aspect' in targets:
            aspect_diff = torch.abs(predictions['aspect'] - targets['aspect'])
            aspect_diff = torch.min(aspect_diff, 360.0 - aspect_diff)
            aspect_loss = aspect_diff.mean()
            losses['aspect'] = aspect_loss
            total_loss = total_loss + self.aspect_weight * aspect_loss
        
        morph_targets = ['curvature', 'tpi', 'tri', 'roughness']
        morph_weight = 0.2
        for name in morph_targets:
            if name in predictions and name in targets:
                loss = F.mse_loss(predictions[name], targets[name])
                losses[name] = loss
                total_loss = total_loss + morph_weight * loss
        
        if 'canopy' in predictions and 'canopy' in targets:
            canopy_mask = targets['canopy'] > 0
            if canopy_mask.sum() > 0:
                canopy_loss = F.mse_loss(
                    predictions['canopy'][canopy_mask], 
                    targets['canopy'][canopy_mask]
                )
                losses['canopy'] = canopy_loss
                total_loss = total_loss + 0.3 * canopy_loss
        
        spectral_targets = ['flow_accum', 'ndvi', 'ndwi', 'bsi', 'shadow']
        spectral_weight = 0.2
        for name in spectral_targets:
            if name in predictions and name in targets:
                loss = F.mse_loss(predictions[name], targets[name])
                losses[name] = loss
                total_loss = total_loss + spectral_weight * loss

        
        if effective_consistency_weight > 0:
            spatial_consistency = self.consistency_loss.spatial_consistency_loss(
                predictions['elevation'], data.edge_index, 
                getattr(data, 'edge_attr', None)
            )
            losses['spatial_consistency'] = spatial_consistency
            losses['effective_consistency_weight'] = effective_consistency_weight
            total_loss = total_loss + effective_consistency_weight * spatial_consistency
        
        if self.terrain_weight > 0:
            elev_mean = getattr(data, 'elev_mean', 0.0)
            elev_std = getattr(data, 'elev_std', 1.0)
            
            terrain_consistency = self.terrain_loss.terrain_consistency_loss(
                predictions['elevation'], 
                predictions.get('slope', torch.zeros_like(predictions['elevation'])),
                predictions.get('aspect', torch.zeros_like(predictions['elevation'])),
                data.edge_index,
                getattr(data, 'coords', None),
                elev_mean=elev_mean,
                elev_std=elev_std
            )
            losses['terrain_consistency'] = terrain_consistency
            total_loss = total_loss + self.terrain_weight * terrain_consistency
            
        if self.sparse_weight > 0 and hasattr(data, 'sparse_mask') and data.sparse_mask is not None:
            anchor_loss = self.sparse_loss(
                predictions['elevation'],
                data.sparse_y,
                data.sparse_mask,
                weight=1.0
            )
            losses['sparse_anchor'] = anchor_loss
            total_loss = total_loss + self.sparse_weight * anchor_loss

        if hasattr(self, 'adaptive_loss_module') and effective_consistency_weight > 0:
             adaptive_loss, adaptive_components = self.adaptive_loss_module(
                 predictions['elevation'],
                 targets['elevation'],
                 targets.get('slope', None),
                 coarse_anchor
             )
             losses['adaptive_total'] = adaptive_loss
             losses['adaptive_anchor'] = adaptive_components['anchor_loss']
             
             total_loss = total_loss + adaptive_loss

        if data.x.size(1) > 14 and effective_consistency_weight > 0:
            water_mask = data.x[:, 14:15]
            pass

        if effective_consistency_weight > 0:
             pass

        losses['total'] = total_loss
        return losses

class AdaptiveGeomorphLoss(nn.Module):
    def __init__(self, 
                 slope_threshold: float = 5.0,
                 smoothness_weight_flat: float = 1.0,
                 smoothness_weight_steep: float = 0.0,
                 anchor_weight_flat: float = 2.0,
                 anchor_weight_steep: float = 0.1):
        super().__init__()
        self.slope_threshold = slope_threshold
        self.smoothness_weight_flat = smoothness_weight_flat
        self.smoothness_weight_steep = smoothness_weight_steep
        self.anchor_weight_flat = anchor_weight_flat
        self.anchor_weight_steep = anchor_weight_steep

    def forward(self, 
                pred_elev: torch.Tensor, 
                target_elev: torch.Tensor,
                target_slope: Optional[torch.Tensor],
                coarse_anchor: Optional[torch.Tensor]):
        
        
        if target_slope is None:
            return torch.tensor(0.0, device=pred_elev.device), {'anchor_loss': 0.0}
            
        if target_slope.max() <= 1.0:
            slope_deg = target_slope * 90.0
        elif target_slope.max() < 2.0:
            slope_deg = target_slope * 180.0 / 3.14159
        else:
            slope_deg = target_slope
            
        is_flat = (slope_deg < self.slope_threshold).float()
        is_steep = 1.0 - is_flat
        
        anchor_loss = torch.tensor(0.0, device=pred_elev.device)
        if coarse_anchor is not None:
            
            diff_anchor = (pred_elev - coarse_anchor).abs()
            
            weighted_diff = (diff_anchor * is_flat * self.anchor_weight_flat) + \
                            (diff_anchor * is_steep * self.anchor_weight_steep)
                            
            anchor_loss = weighted_diff.mean()

        
        
        return anchor_loss, {'anchor_loss': anchor_loss.item()}


class SpectralConsistencyLoss(nn.Module):
    
    SENTINEL_INDICES = [8, 9, 10, 11]
    NDVI_INDEX = 12
    
    def __init__(self, weight: float = 0.3, use_ndvi_only: bool = False):
        super().__init__()
        self.weight = weight
        self.use_ndvi_only = use_ndvi_only
        logger.info(f"SpectralConsistencyLoss initialized (weight={weight}, ndvi_only={use_ndvi_only})")
    
    def forward(self, 
                pred_elev: torch.Tensor, 
                data,
                return_metrics: bool = False) -> torch.Tensor:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            zero_loss = torch.tensor(0.0, device=pred_elev.device, requires_grad=True)
            if return_metrics:
                return zero_loss, {'spectral_consistency': 0.0}
            return zero_loss
        
        if data.x.size(1) <= max(self.SENTINEL_INDICES):
            logger.warning("SpectralConsistencyLoss: Insufficient features for spectral analysis")
            zero_loss = torch.tensor(0.0, device=pred_elev.device, requires_grad=True)
            if return_metrics:
                return zero_loss, {'spectral_consistency': 0.0}
            return zero_loss
        
        row, col = data.edge_index
        
        pred_elev = pred_elev.view(-1)
        
        elev_diff = torch.abs(pred_elev[row] - pred_elev[col])
        
        if self.use_ndvi_only:
            ndvi = data.x[:, self.NDVI_INDEX]
            spectral_diff = torch.abs(ndvi[row] - ndvi[col])
        else:
            sentinel_feats = data.x[:, self.SENTINEL_INDICES]
            spectral_diff = torch.abs(sentinel_feats[row] - sentinel_feats[col]).mean(dim=1)
        
        spectral_similarity = (1.0 - spectral_diff).clamp(0.0, 1.0)
        
        loss = (elev_diff * spectral_similarity).mean()
        
        weighted_loss = self.weight * loss
        
        if return_metrics:
            return weighted_loss, {'spectral_consistency': loss.item()}
        return weighted_loss


class AdaptiveSpectralConsistencyLoss(nn.Module):
    
    SENTINEL_INDICES = [8, 9, 10, 11]
    LIDAR_INDEX = 14
    
    def __init__(self, 
                 base_weight: float = 0.3, 
                 no_lidar_multiplier: float = 4.0,
                 use_ndvi_only: bool = False):
        super().__init__()
        self.base_weight = base_weight
        self.no_lidar_multiplier = no_lidar_multiplier
        self.use_ndvi_only = use_ndvi_only
        
        logger.info(f"AdaptiveSpectralConsistencyLoss initialized (base_weight={base_weight}, no_lidar_mult={no_lidar_multiplier})")
    
    def forward(self, 
                pred_elev: torch.Tensor, 
                data,
                return_metrics: bool = False) -> torch.Tensor:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            zero_loss = torch.tensor(0.0, device=pred_elev.device, requires_grad=True)
            if return_metrics:
                return zero_loss, {'spectral_adaptive': 0.0, 'pct_no_lidar_edges': 0.0}
            return zero_loss
        
        if data.x.size(1) <= self.LIDAR_INDEX:
            logger.warning("AdaptiveSpectralConsistencyLoss: Insufficient features")
            zero_loss = torch.tensor(0.0, device=pred_elev.device, requires_grad=True)
            if return_metrics:
                return zero_loss, {'spectral_adaptive': 0.0, 'pct_no_lidar_edges': 0.0}
            return zero_loss
        
        row, col = data.edge_index
        pred_elev = pred_elev.view(-1)
        
        elev_diff = torch.abs(pred_elev[row] - pred_elev[col])
        
        if self.use_ndvi_only:
            ndvi = data.x[:, 12]
            spectral_diff = torch.abs(ndvi[row] - ndvi[col])
        else:
            sentinel_feats = data.x[:, self.SENTINEL_INDICES]
            spectral_diff = torch.abs(sentinel_feats[row] - sentinel_feats[col]).mean(dim=1)
        
        spectral_similarity = (1.0 - spectral_diff).clamp(0.0, 1.0)
        
        has_lidar = data.x[:, self.LIDAR_INDEX]
        
        edge_has_lidar = (has_lidar[row] > 0.5) | (has_lidar[col] > 0.5)
        
        edge_weights = torch.where(
            edge_has_lidar,
            torch.full_like(elev_diff, self.base_weight),
            torch.full_like(elev_diff, self.base_weight * self.no_lidar_multiplier)
        )
        
        per_edge_loss = elev_diff * spectral_similarity * edge_weights
        loss = per_edge_loss.mean()
        
        if return_metrics:
            pct_no_lidar = (~edge_has_lidar).float().mean().item() * 100
            return loss, {
                'spectral_adaptive': loss.item(),
                'pct_no_lidar_edges': round(pct_no_lidar, 1)
            }
        return loss


class SteepSlopeLoss(nn.Module):
    def __init__(self, slope_threshold: float = 20.0, steep_weight: float = 2.0):
        super().__init__()
        self.slope_threshold = slope_threshold
        self.steep_weight = steep_weight
        logger.info(f"SteepSlopeLoss: threshold = {slope_threshold}°, weight = {steep_weight}x")
    
    def forward(self, pred_elev: torch.Tensor, target_elev: torch.Tensor, 
                slope_deg: torch.Tensor, return_metrics: bool = False) -> torch.Tensor:
        pred = pred_elev.view(-1)
        target = target_elev.view(-1)
        slope = slope_deg.view(-1)
        
        steep_mask = slope > self.slope_threshold
        
        if steep_mask.sum() == 0:
            if return_metrics:
                return torch.tensor(0.0, device=pred.device, requires_grad=True), {'steep_slope_loss': 0.0, 'steep_nodes': 0}
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        steep_error = torch.abs(pred[steep_mask] - target[steep_mask])
        loss = steep_error.mean() * self.steep_weight
        
        if return_metrics:
            return loss, {'steep_slope_loss': loss.item(), 'steep_nodes': steep_mask.sum().item()}
        return loss


class WaterFlatteningLoss(nn.Module):
    
    NDVI_INDEX = 12
    NIR_INDEX = 11
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        logger.info(f"WaterFlatteningLoss initialized (weight = {weight})")
        
    def forward(self, pred_elev: torch.Tensor, data, return_metrics: bool = False) -> torch.Tensor:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            return torch.tensor(0.0, device=pred_elev.device, requires_grad=True)
            
        row, col = data.edge_index
        pred_elev = pred_elev.view(-1)
        
        
        if data.y is None or data.y.size(1) <= 4:
            ndvi = data.x[:, 12]
            nir = data.x[:, 11] if data.x.size(1) > 11 else torch.ones_like(ndvi)
            is_water = (ndvi < 0.1) & (nir < 0.15)
        else:
            ndvi = data.x[:, 12]
            nir = data.x[:, 11] if data.x.size(1) > 11 else torch.ones_like(ndvi)
            slope = data.y[:, 1]
            tpi = data.y[:, 4]
            
            is_water = (ndvi < 0.1) & (nir < 0.15) & (slope < 5.0) & (tpi < 0)
            
        edge_is_water = is_water[row] & is_water[col]
        
        if edge_is_water.sum() == 0:
             if return_metrics:
                return torch.tensor(0.0, device=pred_elev.device, requires_grad=True), {'water_flattening': 0.0}
             return torch.tensor(0.0, device=pred_elev.device, requires_grad=True)
        
        elev_diff = torch.abs(pred_elev[row] - pred_elev[col])
        water_loss = (elev_diff * edge_is_water).sum() / (edge_is_water.sum() + 1e-6)
        
        weighted_loss = self.weight * water_loss
        
        if return_metrics:
            return weighted_loss, {'water_flattening': water_loss.item()}
        return weighted_loss


class MultiTaskLossV9(nn.Module):
    
    TARGET_NAMES = ['elevation', 'slope', 'aspect', 'curvature', 'tpi', 'tri', 'roughness', 'flow_accum']
    
    def __init__(self, 
                 elevation_weight: float = 1.0,
                 slope_weight: float = 0.5,
                 aspect_weight: float = 0.3,
                 curvature_weight: float = 0.3,
                 tpi_weight: float = 0.3,
                 tri_weight: float = 0.3,
                 roughness_weight: float = 0.3,
                 flow_accum_weight: float = 0.5,
                 consistency_weight: float = 0.1,
                 terrain_weight: float = 0.1,
                 sparse_weight: float = 50.0,
                 spectral_weight: float = 0.3,
                 water_weight: float = 1.0,
                 steep_slope_weight: float = 1.0,
                 use_huber: bool = True,
                 huber_delta: float = 1.0,
                 use_adaptive_spectral: bool = True,
                 learnable_weights: bool = False,
                 weight_min_clamp: float = 0.1,
                 weight_max_clamp: float = 2.0,
                 **kwargs):
        
        super().__init__()
        
        self.weights = {
            'elevation': elevation_weight,
            'slope': slope_weight,
            'aspect': aspect_weight,
            'curvature': curvature_weight,
            'tpi': tpi_weight,
            'tri': tri_weight,
            'roughness': roughness_weight,
            'flow_accum': flow_accum_weight,
            'consistency': consistency_weight,
            'terrain': terrain_weight,
            'sparse': sparse_weight,
            'spectral': spectral_weight,
            'water': water_weight,
            'steep_slope': steep_slope_weight
        }
        
        self.sparse_weight = sparse_weight
        
        self.use_huber = use_huber
        
        if use_huber:
            self.elevation_loss = nn.HuberLoss(delta=huber_delta)
        else:
            self.elevation_loss = nn.L1Loss()
            
        self.consistency_loss = ConsistencyLoss()
        self.sparse_loss = SparseAnchorLoss()
        self.geomorph_loss = GeomorphLoss()
        self.terrain_loss = TerrainLoss()
        
        if use_adaptive_spectral:
            self.spectral_loss = AdaptiveSpectralConsistencyLoss(base_weight=1.0)
            logger.info("📡 Using AdaptiveSpectralConsistencyLoss")
        elif spectral_weight > 0:
            self.spectral_loss = SpectralConsistencyLoss(weight=1.0)
            logger.info('Using the standard SpectralConsistencyLoss')
        else:
            self.spectral_loss = None
            
        if water_weight > 0:
            self.water_loss = WaterFlatteningLoss(weight=1.0)
            logger.info("💧 Using WaterFlatteningLoss")
        else:
            self.water_loss = None
        
        if steep_slope_weight > 0:
            self.steep_loss = SteepSlopeLoss(slope_threshold=20.0, steep_weight=1.0)
            logger.info("⛰️ Using SteepSlopeLoss")
        else:
            self.steep_loss = None
            
        self.gating_loss = SpectralGatingLoss(weight=1.0)
            
        self.base_weights = self.weights.copy()
        self.learnable_weights = learnable_weights
        if learnable_weights:
            self.weight_deltas = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(0.0)) 
                for name in self.base_weights.keys() 
                if isinstance(self.base_weights[name], (float, int))
            })
        else:
            self.weight_deltas = None

        logger.info(f"MultiTaskLossV9 initialized (elev={elevation_weight}, water={water_weight})")

    def _get_weight(self, name: str) -> torch.Tensor:
        base = self.base_weights.get(name, 1.0)
        return torch.tensor(base, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')

    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                data,
                epoch: Optional[int] = None,
                warmup_epochs: int = 5,
                aux_outputs: Optional[Dict[str, torch.Tensor]] = None) -> tuple:
        metrics = {}
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        pred_elev = predictions[:, 0]
        target_elev = targets[:, 0]
        l_elev = self.elevation_loss(pred_elev, target_elev)
        w_elev = self._get_weight('elevation')
        total_loss += w_elev * l_elev
        metrics['elevation_loss'] = l_elev.item()
        
        with torch.no_grad():
            elev_mae = torch.abs(pred_elev - target_elev).mean()
            elev_rmse = torch.sqrt(F.mse_loss(pred_elev, target_elev))
            metrics['elevation_mae'] = elev_mae
            metrics['elevation_rmse'] = elev_rmse
        
        for idx in range(1, 8):
            name = self.TARGET_NAMES[idx]
            pred = predictions[:, idx]
            target = targets[:, idx]
            w = self._get_weight(name)
            
            if name == 'aspect':
                diff = torch.abs(pred - target)
                diff = diff % 1.0
                diff = torch.min(diff, 1.0 - diff)
                l = diff.mean()
            else:
                l = F.l1_loss(pred, target)
                
            total_loss += w * l
            metrics[f'{name}_loss'] = l.item()
            
            with torch.no_grad():
                if name == 'aspect':
                    diff_metric = torch.abs(pred - target) % 1.0
                    diff_metric = torch.min(diff_metric, 1.0 - diff_metric)
                    mae = diff_metric.mean()
                    metrics[f'{name}_deg_mae'] = mae * 360.0
                else:
                    mae = torch.abs(pred - target).mean()
                    
                rmse = torch.sqrt(F.mse_loss(pred, target))
                metrics[f'{name}_mae'] = mae
                metrics[f'{name}_rmse'] = rmse
            
        if self.spectral_loss is not None:
            w_spec = self._get_weight('spectral')
            l_spec, m_spec = self.spectral_loss(pred_elev, data, return_metrics=True)
            total_loss += w_spec * l_spec
            metrics.update(m_spec)
            
        if self.water_loss is not None:
             w_water = self._get_weight('water')
             l_water, m_water = self.water_loss(pred_elev, data, return_metrics=True)
             total_loss += w_water * l_water
             metrics.update(m_water)
        
        if self.steep_loss is not None and targets.size(1) > 1:
            w_steep = self._get_weight('steep_slope')
            slope_deg = targets[:, 1] * 90.0
            l_steep, m_steep = self.steep_loss(pred_elev, target_elev, slope_deg, return_metrics=True)
            total_loss += w_steep * l_steep
            metrics.update(m_steep)
             
        w_sparse = self._get_weight('sparse')
        
        if data.x.size(1) > 14:
             sparse_mask = data.x[:, 14] > 0.5
             l_sparse = self.sparse_loss(pred_elev, targets[:, 0], sparse_mask)
        else:
             l_sparse = torch.tensor(0.0, device=total_loss.device)
             
        total_loss += w_sparse * l_sparse
        metrics['sparse_loss'] = l_sparse.item()
        
        if aux_outputs and 'aux_canopy_height' in aux_outputs:
            canopy_pred = aux_outputs['aux_canopy_height']
            w_gate = self._get_weight('spectral') 
            l_gate, m_gate = self.gating_loss(canopy_pred, data.x)
            
            total_loss += w_gate * l_gate
            metrics.update(m_gate)
        
        w_consistency = self._get_weight('consistency')
        w_terrain = self._get_weight('terrain')
        
        if w_consistency > 0:
            spatial_loss = self.consistency_loss.spatial_consistency_loss(
                predictions[:, 0:1],
                data.edge_index,
                getattr(data, 'edge_attr', None)
            )
            metrics['consistency'] = spatial_loss.item()
            total_loss = total_loss + w_consistency * spatial_loss
            
            elev_mean = getattr(data, 'elev_mean', 0.0)
            elev_std = getattr(data, 'elev_std', 1.0)
            
            terrain_consistency = self.terrain_loss.terrain_consistency_loss(
                predictions[:, 0],
                predictions[:, 1],
                predictions[:, 2],
                data.edge_index,
                getattr(data, 'pos', None) if hasattr(data, 'pos') else getattr(data, 'coords', None),
                elev_mean=elev_mean,
                elev_std=elev_std
            )
            metrics['terrain'] = terrain_consistency.item()
            total_loss = total_loss + w_terrain * terrain_consistency

            geomorph_loss_val, geomorph_components = self.geomorph_loss(
                predictions[:, 0],
                predictions[:, 1],
                data.edge_index,
                data
            )
            metrics['geomorph'] = geomorph_loss_val.item()
            total_loss = total_loss + w_consistency * geomorph_loss_val

        metrics['total'] = total_loss
        return metrics
    
    def get_target_weights(self) -> Dict[str, float]:
        return self.weights.copy()

