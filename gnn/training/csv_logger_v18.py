import csv
import os
import torch
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CSVLoggerV18:
    def __init__(self, save_dir: str, filename: str = "training_log.csv"):
        self.filepath = os.path.join(save_dir, filename)
        self.headers = [
            'epoch', 'phase', 'train_total', 'val_total', 'lr_current', 'grad_norm', 'gpu_memory_gb',
            'loss_delta', 'overfit_ratio', 'best_epoch',
            'physics_consistency', 'physics_terrain', 'physics_water_flow', 'physics_river_width', 'physics_geomorph',
            'sparse_anchor_loss', 'sparse_weight', 'anchor_mae', 'non_anchor_mae',
            'pred_mean', 'pred_std', 'error_p90',
            'frozen_backbone_pct', 'frozen_heads_count',
            'corr_b02_error', 'corr_b03_error', 'corr_b04_error', 'corr_b08_error', 'corr_has_lidar_error',
            'corr_edge_diff_b04_error', 'corr_edge_diff_b08_error', 'corr_edge_diff_ndvi_error',
            'corr_edge_anchor_prox_error', 'corr_edge_shadow_diff_error', 'corr_edge_both_anchors_error',
            'p90_water', 'p90_dense_veg', 'p90_clearing', 'p90_steep_slope', 'p90_gentle_slope',
            'p90_peak_extreme', 'p90_peak_moderate', 'p90_valley_extreme', 'p90_valley_moderate',
            'p90_with_lidar', 'p90_no_lidar',
            'spectral_adaptive_loss', 'water_flattening_loss', 'steep_slope_loss',
            'mae_lidar', 'mae_no_lidar', 'gap_ratio',
            'water_nodes_count', 'water_elev_std',
            'canopy_mean', 'canopy_max', 'corr_ndvi_canopy',
            'canopy_forest_mean', 'canopy_open_mean', 'canopy_water_mean', 'canopy_rock_mean',
            'spectral_gating_loss', 'gating_efficiency', 'false_correction_rate', 'spectral_bias_ratio',
            'elevation_loss_train', 'elevation_loss_val', 'elevation_mae', 'elevation_rmse', 'elevation_weight', 'elevation_status',
            'slope_loss_train', 'slope_loss_val', 'slope_mae', 'slope_rmse', 'slope_weight', 'slope_status',
            'aspect_deg_loss_train', 'aspect_deg_loss_val', 'aspect_deg_mae', 'aspect_deg_rmse', 'aspect_deg_weight', 'aspect_deg_status',
            'curvature_loss_train', 'curvature_loss_val', 'curvature_mae', 'curvature_rmse', 'curvature_weight', 'curvature_status',
            'tpi_loss_train', 'tpi_loss_val', 'tpi_mae', 'tpi_rmse', 'tpi_weight', 'tpi_status',
            'tri_loss_train', 'tri_loss_val', 'tri_mae', 'tri_rmse', 'tri_weight', 'tri_status',
            'roughness_loss_train', 'roughness_loss_val', 'roughness_mae', 'roughness_rmse', 'roughness_weight', 'roughness_status',
            'flow_accum_loss_train', 'flow_accum_loss_val', 'flow_accum_mae', 'flow_accum_rmse', 'flow_accum_weight', 'flow_accum_status'
        ]
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def compute_correlations(self, error, features):
        corrs = []
        if features is None or features.shape[0] == 0:
            return [0.0] * features.shape[1] if features is not None else []
            
        error_centered = error - error.mean()
        for i in range(features.shape[1]):
            feat = features[:, i]
            feat_centered = feat - feat.mean()
            numerator = (error_centered * feat_centered).sum()
            denominator = torch.sqrt((error_centered**2).sum()) * torch.sqrt((feat_centered**2).sum()) + 1e-8
            corrs.append((numerator / denominator).item())
        return corrs

    def compute_p90_by_mask(self, error, mask):
        if mask is None or mask.sum() == 0:
            return 0.0
        subset_error = error[mask]
        return torch.quantile(subset_error, 0.9).item()

    def log_epoch(self, epoch_data: dict):
        row = []
        for h in self.headers:
            val = epoch_data.get(h, 0.0)
            if isinstance(val, (float, np.floating)):
                val = f"{val:.6f}"
            row.append(val)
            
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        
        logger.info(f"Logged epoch to {self.filepath}")


    def calculate_derived_metrics(self, preds, targets, sentinel_feats=None, dem_feats=None, canopy_preds=None, stats=None):
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()
        
        min_len = min(preds.size(0), targets.size(0))
        
        if dem_feats is not None: 
            if dem_feats.size(0) > min_len:
                print(f"Truncating features from {dem_feats.size(0)} to {min_len}", flush=True)
            min_len = min(min_len, dem_feats.size(0))
            
        if sentinel_feats is not None: min_len = min(min_len, sentinel_feats.size(0))
        if canopy_preds is not None: min_len = min(min_len, canopy_preds.numel())
        
        preds = preds[:min_len]
        targets = targets[:min_len]
        if dem_feats is not None: dem_feats = dem_feats[:min_len]
        if sentinel_feats is not None: sentinel_feats = sentinel_feats[:min_len]
        if canopy_preds is not None: canopy_preds = canopy_preds.view(-1)[:min_len]
        
        metrics = {}
        
        error = torch.abs(preds - targets)
        n_samples = error.size(0)
        metrics['pred_mean'] = preds.mean().item()
        metrics['pred_std'] = preds.std().item()
        metrics['error_p90'] = torch.quantile(error, 0.9).item() if len(error) > 0 else 0.0
        
        if dem_feats is not None and dem_feats.shape[1] >= 12 and dem_feats.size(0) == n_samples:
            feat_subset = dem_feats[:, 8:12].float()
            corrs = self.compute_correlations(error, feat_subset)
            metrics['corr_b02_error'] = corrs[0] if len(corrs) > 0 else 0.0
            metrics['corr_b03_error'] = corrs[1] if len(corrs) > 1 else 0.0
            metrics['corr_b04_error'] = corrs[2] if len(corrs) > 2 else 0.0
            metrics['corr_b08_error'] = corrs[3] if len(corrs) > 3 else 0.0
            
            if dem_feats.shape[1] >= 13:
                ndvi_corr = self.compute_correlations(error, dem_feats[:, 12:13].float())
                metrics['corr_ndvi_canopy'] = ndvi_corr[0] if len(ndvi_corr) > 0 else 0.0
        else:
            metrics['corr_b02_error'] = 0.0
            metrics['corr_b03_error'] = 0.0
            metrics['corr_b04_error'] = 0.0
            metrics['corr_b08_error'] = 0.0
            metrics['corr_ndvi_canopy'] = 0.0
        
        if dem_feats is not None:
            n_feats = dem_feats.shape[1]
            
            if n_feats > 1:
                slope = dem_feats[:, 1]
                mask_steep = slope > 0.33
                mask_gentle = slope < 0.05
                metrics['p90_steep_slope'] = self.compute_p90_by_mask(error, mask_steep)
                metrics['p90_gentle_slope'] = self.compute_p90_by_mask(error, mask_gentle)
            
            if n_feats > 12:
                ndvi = dem_feats[:, 12]
                mask_dense_veg = ndvi > 0.6
                mask_clearing = (ndvi > -0.1) & (ndvi < 0.2)
                metrics['p90_dense_veg'] = self.compute_p90_by_mask(error, mask_dense_veg)
                metrics['p90_clearing'] = self.compute_p90_by_mask(error, mask_clearing)
            
            if n_feats > 13:
                ndwi = dem_feats[:, 13]
                mask_water = ndwi > 0.3
                metrics['p90_water'] = self.compute_p90_by_mask(error, mask_water)
                metrics['water_nodes_count'] = int(mask_water.sum().item())
                if mask_water.sum() > 1:
                    metrics['water_elev_std'] = targets.flatten()[mask_water].std().item()
            
            if n_feats > 5:
                tpi = dem_feats[:, 5]
                mask_peak_extreme = tpi > 0.5
                mask_peak_moderate = (tpi > 0.2) & (tpi <= 0.5)
                mask_valley_extreme = tpi < -0.5
                mask_valley_moderate = (tpi >= -0.5) & (tpi < -0.2)
                metrics['p90_peak_extreme'] = self.compute_p90_by_mask(error, mask_peak_extreme)
                metrics['p90_peak_moderate'] = self.compute_p90_by_mask(error, mask_peak_moderate)
                metrics['p90_valley_extreme'] = self.compute_p90_by_mask(error, mask_valley_extreme)
                metrics['p90_valley_moderate'] = self.compute_p90_by_mask(error, mask_valley_moderate)
            
            if n_feats > 14:
                has_lidar = dem_feats[:, 14] > 0.5
                no_lidar = ~has_lidar
                metrics['p90_with_lidar'] = self.compute_p90_by_mask(error, has_lidar)
                metrics['p90_no_lidar'] = self.compute_p90_by_mask(error, no_lidar)
                metrics['corr_has_lidar_error'] = self.compute_correlations(error, dem_feats[:, 14:15])[0] if len(error) > 0 else 0.0
                
                if has_lidar.sum() > 0:
                    metrics['anchor_mae'] = error[has_lidar].mean().item()
                    metrics['mae_lidar'] = metrics['anchor_mae']
                if no_lidar.sum() > 0:
                    metrics['non_anchor_mae'] = error[no_lidar].mean().item()
                    metrics['mae_no_lidar'] = metrics['non_anchor_mae']
                if has_lidar.sum() > 0 and no_lidar.sum() > 0:
                    metrics['gap_ratio'] = metrics['non_anchor_mae'] / (metrics['anchor_mae'] + 1e-6)
            
            
            pass
        
        MAX_CANOPY_HEIGHT = 50.0
        
        if canopy_preds is not None and canopy_preds.numel() > 0:
            canopy_h_m = canopy_preds.flatten() * MAX_CANOPY_HEIGHT
            
            metrics['canopy_mean'] = canopy_h_m.mean().item()
            metrics['canopy_max'] = canopy_h_m.max().item()
            
            if dem_feats is not None and dem_feats.shape[1] > 12:
                ndvi = dem_feats[:, 12]
                ndwi = dem_feats[:, 13] if dem_feats.shape[1] > 13 else None
                
                mask_forest = ndvi > 0.6
                mask_open = ndvi < 0.2
                mask_water = (ndwi > 0.1) if ndwi is not None else (ndvi < -0.1)
                mask_rock = (ndvi < 0.2) & (~mask_water if ndwi is not None else True)
                
                if mask_forest.sum() > 0:
                    metrics['canopy_forest_mean'] = canopy_h_m[mask_forest].mean().item()
                if mask_open.sum() > 0:
                    metrics['canopy_open_mean'] = canopy_h_m[mask_open].mean().item()
                if mask_water.sum() > 0:
                    metrics['canopy_water_mean'] = canopy_h_m[mask_water].mean().item()
                if mask_rock.sum() > 0:
                    metrics['canopy_rock_mean'] = canopy_h_m[mask_rock].mean().item()
                
                if canopy_h_m.std() > 1e-6 and ndvi.std() > 1e-6:
                    corr = self.compute_correlations(canopy_h_m, ndvi.unsqueeze(1))
                    metrics['corr_ndvi_canopy'] = corr[0] if len(corr) > 0 else 0.0
        else:
            metrics['canopy_mean'] = 0.0
            metrics['canopy_max'] = 0.0
            metrics['canopy_forest_mean'] = 0.0
            metrics['canopy_open_mean'] = 0.0
            metrics['canopy_water_mean'] = 0.0
            metrics['canopy_rock_mean'] = 0.0
            metrics['corr_ndvi_canopy'] = 0.0

        
        if sentinel_feats is not None and sentinel_feats.shape[0] > 1:
            error_diff = torch.abs(error[1:] - error[:-1])
            if sentinel_feats.shape[1] >= 4:
                b04_diff = torch.abs(sentinel_feats[1:, 2] - sentinel_feats[:-1, 2])
                b08_diff = torch.abs(sentinel_feats[1:, 3] - sentinel_feats[:-1, 3])
                corr_edge_b04 = self.compute_correlations(error_diff, b04_diff.unsqueeze(1))
                corr_edge_b08 = self.compute_correlations(error_diff, b08_diff.unsqueeze(1))
                metrics['corr_edge_diff_b04_error'] = corr_edge_b04[0] if len(corr_edge_b04) > 0 else 0.0
                metrics['corr_edge_diff_b08_error'] = corr_edge_b08[0] if len(corr_edge_b08) > 0 else 0.0
                
            if sentinel_feats.shape[1] >= 5:
                ndvi_diff = torch.abs(sentinel_feats[1:, 4] - sentinel_feats[:-1, 4])
                corr_edge_ndvi = self.compute_correlations(error_diff, ndvi_diff.unsqueeze(1))
                metrics['corr_edge_diff_ndvi_error'] = corr_edge_ndvi[0] if len(corr_edge_ndvi) > 0 else 0.0
        
        if dem_feats is not None and dem_feats.shape[1] > 14 and dem_feats.shape[0] > 1:
            has_lidar = dem_feats[:, 14]
            anchor_prox = has_lidar[1:] + has_lidar[:-1]
            error_diff = torch.abs(error[1:] - error[:-1])
            corr_anchor_prox = self.compute_correlations(error_diff, anchor_prox.unsqueeze(1))
            metrics['corr_edge_anchor_prox_error'] = corr_anchor_prox[0] if len(corr_anchor_prox) > 0 else 0.0
            
            both_anchors = (has_lidar[1:] > 0.5) & (has_lidar[:-1] > 0.5)
            metrics['corr_edge_both_anchors_error'] = self.compute_p90_by_mask(error_diff, both_anchors)
        
        if sentinel_feats is not None and sentinel_feats.shape[1] >= 3 and sentinel_feats.shape[0] > 1:
            b04 = sentinel_feats[:, 2]
            is_shadow = b04 < 0.1
            shadow_diff = (is_shadow[1:].float() - is_shadow[:-1].float()).abs()
            error_diff = torch.abs(error[1:] - error[:-1])
            corr_shadow = self.compute_correlations(error_diff, shadow_diff.unsqueeze(1))
            metrics['corr_edge_shadow_diff_error'] = corr_shadow[0] if len(corr_shadow) > 0 else 0.0
        
        
        if dem_feats is not None:
            n_feats = dem_feats.shape[1]
            
            if n_feats > 13:
                ndwi = dem_feats[:, 13]
                mask_water = ndwi > 0.3
                if mask_water.sum() > 10:
                    water_preds = preds.flatten()[mask_water]
                    water_std = water_preds.std().item()
                    water_range = (water_preds.max() - water_preds.min()).item()
                    metrics['physics_water_flow'] = water_std
                    metrics['water_flattening_loss'] = water_range
            
            if n_feats > 1:
                dem_slope = dem_feats[:, 1]
                gentle_mask = dem_slope < 0.1
                steep_mask = dem_slope > 0.4
                if gentle_mask.sum() > 0 and steep_mask.sum() > 0:
                    gentle_mae = error[gentle_mask].mean().item()
                    steep_mae = error[steep_mask].mean().item()
                    metrics['physics_terrain'] = steep_mae - gentle_mae
                    metrics['steep_slope_loss'] = steep_mae
            
            if n_feats > 5:
                tpi = dem_feats[:, 5]
                mask_peak = tpi > 0.3
                mask_valley = tpi < -0.3
                mask_flat = torch.abs(tpi) < 0.1
                
                peak_err = error[mask_peak].mean().item() if mask_peak.sum() > 0 else 0
                valley_err = error[mask_valley].mean().item() if mask_valley.sum() > 0 else 0
                flat_err = error[mask_flat].mean().item() if mask_flat.sum() > 0 else 0
                
                metrics['physics_geomorph'] = max(peak_err, valley_err) - flat_err
            
            if n_feats > 13 and dem_feats.shape[0] > 1:
                ndwi = dem_feats[:, 13]
                ndwi_grad = torch.abs(ndwi[1:] - ndwi[:-1])
                water_edge = ndwi_grad > 0.2
                if water_edge.sum() > 10:
                    edge_error = error[1:][water_edge].mean().item()
                    metrics['physics_river_width'] = edge_error
            
            metrics['physics_consistency'] = (
                metrics.get('physics_terrain', 0) * 0.3 +
                metrics.get('physics_water_flow', 0) * 0.3 +
                metrics.get('physics_geomorph', 0) * 0.4
            )
        
        
        if sentinel_feats is not None:
            n_bands = sentinel_feats.shape[1]
            
            if n_bands >= 4:
                b04 = sentinel_feats[:, 2]
                b08 = sentinel_feats[:, 3]
                veg_index = (b08 - b04) / (b08 + b04 + 1e-6)
                
                high_conf = torch.abs(veg_index) > 0.3
                if high_conf.sum() > 0:
                    metrics['spectral_adaptive_loss'] = error[high_conf].mean().item()
            
            if n_bands >= 5:
                ndvi = sentinel_feats[:, 4]
                high_ndvi = ndvi > 0.5
                low_ndvi = ndvi < 0.2
                
                if high_ndvi.sum() > 0 and low_ndvi.sum() > 0:
                    veg_err = error[high_ndvi].mean().item()
                    bare_err = error[low_ndvi].mean().item()
                    metrics['spectral_gating_loss'] = abs(veg_err - bare_err)
                    metrics['gating_efficiency'] = bare_err / (veg_err + 1e-6) if veg_err > 0 else 0
            
            if n_bands >= 4:
                bright = sentinel_feats[:, 2] > 0.3
                if bright.sum() > 0:
                    bright_p90 = torch.quantile(error[bright], 0.9).item() if bright.sum() > 10 else 0
                    overall_p90 = metrics.get('error_p90', 0)
                    metrics['false_correction_rate'] = bright_p90 / (overall_p90 + 1e-6) if overall_p90 > 0 else 0
            
            if n_bands >= 4:
                green_ratio = sentinel_feats[:, 3] / (sentinel_feats[:, 2] + 1e-6)
                green = green_ratio > 2.0
                non_green = green_ratio < 1.5
                if green.sum() > 0 and non_green.sum() > 0:
                    green_mae = error[green].mean().item()
                    non_green_mae = error[non_green].mean().item()
                    metrics['spectral_bias_ratio'] = green_mae / (non_green_mae + 1e-6)
        
        if dem_feats is not None and dem_feats.shape[1] > 14:
            has_lidar = dem_feats[:, 14] > 0.5
            anchor_count = has_lidar.sum().item()
            total_count = len(has_lidar)
            
            metrics['sparse_weight'] = anchor_count / (total_count + 1e-6)
            
            if anchor_count > 0:
                anchor_err = error[has_lidar]
                metrics['sparse_anchor_loss'] = anchor_err.mean().item()
        
        return metrics


