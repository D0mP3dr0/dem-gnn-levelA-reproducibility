import pytorch_lightning as pl
import torch
import torch.optim as optim
from typing import Dict, Any, Optional
import logging

from models.hierarchical_gnn import HierarchicalTerrainGNN
from models.hetero_losses import MultiTaskLossHetero
from config import PHASE_CONFIGS, NORMALIZATION_STATS, TARGET_INDICES
from training.csv_logger_v18 import CSVLoggerV18

logger = logging.getLogger(__name__)

class HierarchicalTrainerModule(pl.LightningModule):
    
    def __init__(self, 
                 metadata: Any,
                 model_config: Dict[str, Any] = None,
                 loss_config: Dict[str, Any] = None,
                 optimizer_config: Dict[str, Any] = None,
                 phase: int = 1):
        super().__init__()
        self.save_hyperparameters()
        
        self.metadata = metadata
        self.model = HierarchicalTerrainGNN(metadata, **(model_config or {}))
        
        self.criterion = MultiTaskLossHetero(**(loss_config or {}))
        
        self.optimizer_config = optimizer_config or {'lr': 1e-4, 'weight_decay': 1e-5}
        
        self.current_phase = phase
        self.phase_config = {}
        
        self.csv_logger = None
        
        self.optimizer_config = optimizer_config or {'lr': 1e-4, 'weight_decay': 1e-5}
        
        self.current_phase = phase
        self.phase_config = PHASE_CONFIGS.get(f"{phase}_Elevation_Foundation", {})
        
        for key in PHASE_CONFIGS:
            if key.startswith(f"{phase}_"):
                self.phase_config = PHASE_CONFIGS[key]
                break
                
        self.update_phase(phase)
        
    def forward(self, batch):
        return self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    
    def on_train_start(self):
        self.train_losses = {}
        self.train_batch_count = 0
        self.epoch_grad_norm = 0.0
        
    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        
        loss_dict = self.criterion(predictions, batch, epoch=self.current_epoch)
        
        self.train_batch_count += 1
        for k, v in loss_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self.train_losses[k] = self.train_losses.get(k, 0.0) + val
        
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.epoch_grad_norm += total_norm ** 0.5
        
        self.log('train_loss', loss_dict['total'], on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        
        for k, v in loss_dict.items():
            if k != 'total':
                self.log(f'train_{k}', v, on_step=False, on_epoch=True, batch_size=1)
                
        if batch_idx % 70 == 0:
             logger.info(f"Phase {self.current_phase} | Ep {self.current_epoch} | Loss: {loss_dict['total']:.4f}")
             
        return loss_dict['total']
    
    def on_train_epoch_end(self):
        if self.train_batch_count > 0:
            self.avg_train_losses = {k: v / self.train_batch_count for k, v in self.train_losses.items()}
            self.avg_grad_norm = self.epoch_grad_norm / self.train_batch_count
        else:
            self.avg_train_losses = {}
            self.avg_grad_norm = 0.0
        
        self.train_losses = {}
        self.train_batch_count = 0
        self.epoch_grad_norm = 0.0

        
    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss_dict = self.criterion(predictions, batch, epoch=self.current_epoch)
        
        self.log('val_loss', loss_dict['total'], on_epoch=True, prog_bar=True, batch_size=1)
        
        
        targets = {}
        if hasattr(batch['dem'], 'y') and batch['dem'].y is not None:
            targets['elevation'] = batch['dem'].y[:, 0]
            if batch['dem'].y.size(1) > 1: targets['slope'] = batch['dem'].y[:, 1]
            if batch['dem'].y.size(1) > 2: targets['aspect_deg'] = batch['dem'].y[:, 2]
            
        return {
            'preds': predictions,
            'targets': targets
        }

    def on_validation_epoch_end(self):
        outputs = self.trainer.callback_metrics
        pass 

    def on_validation_start(self):
        self.val_outputs = []
        if self.csv_logger is None and self.trainer:
             log_dir = self.trainer.logger.log_dir or self.trainer.default_root_dir
             self.csv_logger = CSVLoggerV18(log_dir)

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss_dict = self.criterion(predictions, batch, epoch=self.current_epoch)
        self.log('val_loss', loss_dict['total'], on_epoch=True, prog_bar=True, batch_size=1)
        
        targets = {}
        if hasattr(batch['dem'], 'y') and batch['dem'].y is not None:
            y = batch['dem'].y
            for name, idx in TARGET_INDICES.items():
                if y.size(1) > idx:
                    targets[name] = y[:, idx]
             
        sentinel_feats = batch['sentinel'].x.detach().cpu() if 'sentinel' in batch.node_types else None
        dem_feats = batch['dem'].x.detach().cpu() if 'dem' in batch.node_types else None
        
        self.val_outputs.append({
            'preds': {k: v.detach().cpu() for k, v in predictions.items()},
            'targets': {k: v.detach().cpu() for k, v in targets.items()},
            'sentinel': sentinel_feats,
            'dem': dem_feats,
            'loss_dict': {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        })
        return loss_dict['total']

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if not hasattr(self, 'val_outputs') or not self.val_outputs:
            return

        if getattr(self.trainer, "sanity_checking", False):
            self.val_outputs = [] 
            return

        if self.csv_logger is None and self.trainer:
             log_dir = self.trainer.logger.log_dir or self.trainer.default_root_dir
             self.csv_logger = CSVLoggerV18(log_dir)

        agg_preds = {}
        agg_targets = {}
        agg_sentinel = []
        agg_dem = []
        val_losses = {} 
        
        count = len(self.val_outputs)
        
        for batch_out in self.val_outputs:
            for k, v in batch_out['loss_dict'].items():
                val_losses[k] = val_losses.get(k, 0.0) + v

            for k, v in batch_out['preds'].items():
                if k not in agg_preds: agg_preds[k] = []
                agg_preds[k].append(v)
            
            for k, v in batch_out['targets'].items():
                if k not in agg_targets: agg_targets[k] = []
                agg_targets[k].append(v)
                
            if batch_out['sentinel'] is not None: agg_sentinel.append(batch_out['sentinel'])
            if batch_out['dem'] is not None: agg_dem.append(batch_out['dem'])
        
        for k in val_losses: val_losses[k] /= count

        for k in agg_preds: agg_preds[k] = torch.cat(agg_preds[k], dim=0)
        for k in agg_targets: agg_targets[k] = torch.cat(agg_targets[k], dim=0)
        
        sentinel_cat = torch.cat(agg_sentinel, dim=0) if agg_sentinel else None
        dem_cat = torch.cat(agg_dem, dim=0) if agg_dem else None
        
        stats = NORMALIZATION_STATS.get(self.hparams.get('biome', 'mata_atlantica'), NORMALIZATION_STATS['default'])
        elev_mean = stats['elev_mean']
        elev_std = stats['elev_std']
        
        phase_weights = self.phase_config.get('weights', {})
        
        lr = 0.0
        try:
            opts = self.optimizers()
            if isinstance(opts, list): opts = opts[0]
            if opts and hasattr(opts, 'param_groups'):
               lr = opts.param_groups[0]['lr']
        except: pass
        
        epoch_data = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'lr_current': lr,
            'val_total': val_losses.get('total', 0.0)
        }
        
        phase_names = {1: '1_Elevation_Foundation', 2: '2_Slope_Physics', 3: '3_Aspect_Orientation', 
                       4: '4_Curvature_Topology', 5: '5_Morphology_Final'}
        phase_name = phase_names.get(self.current_phase, f'Phase_{self.current_phase}')
        
        denorm_configs = [
            ('elevation', elev_std, elev_mean, 'm'),
            ('slope', 90.0, 0.0, '°'),
            ('aspect_cos', 1.0, 0.0, ''),
            ('aspect_sin', 1.0, 0.0, ''),
            ('curvature', 1.0, 0.0, ''),
            ('tpi', elev_std, 0.0, 'm'),
            ('tri', elev_std, 0.0, 'm'),
            ('roughness', elev_std, 0.0, 'm'),
            ('flow_accum', 1.0, 0.0, ''),
            ('canopy', 50.0, 0.0, 'm'),
            ('ndvi', 1.0, 0.0, ''),
            ('ndwi', 1.0, 0.0, ''),
            ('bsi', 1.0, 0.0, ''),
            ('shadow', 1.0, 0.0, ''),
        ]
        
        torch.cuda.empty_cache()
        
        print(f"\n[{phase_name}] Ep {self.current_epoch}/{self.trainer.max_epochs} | Val Loss: {val_losses.get('total', 0.0):.2f}", flush=True)
        print(f"{'Target':<15} {'Val MAE':<10} {'Val RMSE':<10} {'GT Mean':<12}", flush=True)
        print("-" * 60, flush=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for target_name, scale, offset, unit in denorm_configs:
            if target_name in agg_preds:
                p_flat = agg_preds[target_name].view(-1).float()
                
                if target_name in agg_targets:
                    t_flat = agg_targets[target_name].view(-1).float()
                else:
                    t_flat = p_flat.clone()
                
                min_len = min(p_flat.size(0), t_flat.size(0))
                p_gpu = (p_flat[:min_len].to(device) * scale + offset)
                t_gpu = (t_flat[:min_len].to(device) * scale + offset)
                
                mae = torch.abs(p_gpu - t_gpu).mean().item()
                rmse = torch.sqrt(torch.mean((p_gpu - t_gpu)**2)).item()
                gt_mean = t_gpu.mean().item()
                
                del p_gpu, t_gpu, p_flat, t_flat
                torch.cuda.empty_cache()
                
                weight = phase_weights.get(target_name, 0.0)
                if weight == 0.0 and 'aspect' in target_name:
                    weight = phase_weights.get('aspect', 0.0)
                status = '[ACTIVE]' if weight > 0 else '[WARMUP]'
                
                epoch_data[f'{target_name}_mae'] = mae
                epoch_data[f'{target_name}_rmse'] = rmse
                epoch_data[f'{target_name}_loss_val'] = val_losses.get(target_name, 0.0)
                
                print(f"{target_name:<15} {mae:>7.2f}{unit:<3} {rmse:>7.2f}{unit:<3} {gt_mean:>8.2f}{unit:<3} {status}", flush=True)
            else:
                print(f"{target_name:<15} {'---':<10} {'---':<10} {'---':<12} [INACTIVE]", flush=True)

        print("-" * 60, flush=True)
        print(f"   📊 LR: {lr:.2e}", flush=True)
        
        torch.cuda.empty_cache()
        
        
        
        all_targets = ['elevation', 'slope', 'aspect_cos', 'aspect_sin', 'curvature', 'tpi', 'tri', 'roughness', 'flow_accum', 'canopy', 'ndvi', 'ndwi', 'bsi', 'shadow']

        if hasattr(self, 'avg_train_losses'):
            epoch_data['train_total'] = self.avg_train_losses.get('total', 0.0)
            for target_name in all_targets:
                epoch_data[f'{target_name}_loss_train'] = self.avg_train_losses.get(target_name, 0.0)
        
        if hasattr(self, 'avg_grad_norm'):
            epoch_data['grad_norm'] = self.avg_grad_norm
        
        if torch.cuda.is_available():
            epoch_data['gpu_memory_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        if hasattr(self, 'prev_val_loss'):
            epoch_data['loss_delta'] = val_losses.get('total', 0.0) - self.prev_val_loss
        else:
            epoch_data['loss_delta'] = 0.0
        self.prev_val_loss = val_losses.get('total', 0.0)
        
        train_total = epoch_data.get('train_total', 0.0)
        val_total = val_losses.get('total', 0.0)
        epoch_data['overfit_ratio'] = train_total / (val_total + 1e-6) if val_total > 0 else 0.0
        
        if not hasattr(self, 'best_val_loss') or val_losses.get('total', 999) < self.best_val_loss:
            self.best_val_loss = val_losses.get('total', 999)
            self.best_epoch = self.current_epoch
        epoch_data['best_epoch'] = getattr(self, 'best_epoch', 0)
        
        frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in self.model.parameters())
        epoch_data['frozen_backbone_pct'] = frozen_params / max(total_params, 1) * 100
        epoch_data['frozen_heads_count'] = 0
        
        for target_name in all_targets:
            weight = phase_weights.get(target_name, 0.0)
            epoch_data[f'{target_name}_weight'] = weight
            epoch_data[f'{target_name}_status'] = 'ACTIVE' if weight > 0 else 'WARMUP'
        
        if 'elevation' in agg_preds and self.csv_logger:
            p_elev = agg_preds['elevation'].float() * elev_std + elev_mean
            t_elev = agg_targets.get('elevation', agg_preds['elevation']).float() * elev_std + elev_mean
            
            canopy_preds = agg_preds.get('canopy', None)
            
            derived = self.csv_logger.calculate_derived_metrics(
                p_elev, t_elev, sentinel_cat, dem_cat, 
                canopy_preds=canopy_preds
            )
            epoch_data.update(derived)
        
        if self.csv_logger:
            self.csv_logger.log_epoch(epoch_data)
        
        self.val_outputs = []
        del agg_preds, agg_targets, agg_sentinel, agg_dem
        torch.cuda.empty_cache()

    def on_test_start(self):
        self.on_validation_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()


        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.optimizer_config.get('lr', 1e-4),
            weight_decay=self.optimizer_config.get('weight_decay', 1e-5)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def update_phase(self, phase: int):
        self.current_phase = phase
        logger.info(f"SWITCHING TO PHASE {phase}...")
        
        self.model.update_phase(phase)
        
        phase_key = None
        for key in PHASE_CONFIGS:
            if key.startswith(f"{phase}_"):
                phase_key = key
                break
        
        if phase_key:
            new_weights = PHASE_CONFIGS[phase_key]['weights']
            logger.info(f"   Updating loss weights: {new_weights}")
            
            target_module = getattr(self.criterion, 'loss_module', self.criterion)
            
            if hasattr(target_module, 'update_weights'):
                target_module.update_weights(new_weights)
            else:
                for name, weight in new_weights.items():
                    attr_name = f"{name}_weight"
                    if hasattr(target_module, attr_name):
                        setattr(target_module, attr_name, weight)
        
        logger.info(f"Phase {phase} configuration applied.")
