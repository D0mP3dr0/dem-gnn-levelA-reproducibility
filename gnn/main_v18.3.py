import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
import logging
import gc
torch.set_float32_matmul_precision('medium')
from pathlib import Path

from config import OUTPUT_DIR, CHECKPOINT_DIR, LOGS_DIR, BIOME_DATASETS, MODEL_CONFIG, HETERO_CONFIG
from data.datamodule_v18 import HierarchicalDataModule
from training.module_v18 import HierarchicalTrainerModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("V18.3_Main")

def run_training_pipeline(biome: str = 'amazonia', dataset_path: str = None, num_workers: int = 0, batch_size_arg: int = 1, limit_fraction: float = 1.0, test_mode: bool = False, chunk_size: int = 350, start_phase: int = 1, ckpt_path: str = None):
    logger.info(f"Starting V18.3 Training Pipeline for {biome.upper()}")
    biome = biome.lower()
    
    if dataset_path is None:
        dataset_path = BIOME_DATASETS.get(biome)
        
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
        
    logger.info(f"   Config: Workers={num_workers} | Batch={batch_size_arg} | Limit={limit_fraction:.4f}")
    if start_phase > 1:
        logger.info(f"   Resuming from Phase {start_phase}")
        if ckpt_path:
             logger.info(f"   📂 Checkpoint: {ckpt_path}")

    dm = HierarchicalDataModule(
        dataset_path=str(dataset_path),
        chunk_size=chunk_size,
        batch_size=batch_size_arg,
        num_workers=num_workers,
        biome=biome,
        test_mode=test_mode,
        limit_fraction=limit_fraction
    )
    
    dm.setup() 
    metadata = dm.data.metadata()
    
    v18_model_config = {
        'hidden_dim': MODEL_CONFIG.get('hidden_channels', 128),
        'num_layers': MODEL_CONFIG.get('num_layers', 4),
        'heads': MODEL_CONFIG.get('heads', 4),
        'dropout': MODEL_CONFIG.get('dropout', 0.1)
    }
    
    model = HierarchicalTrainerModule(
        metadata=metadata,
        model_config=v18_model_config,
        phase=start_phase
    )
    
    total_phases = 5
    
    PHASE_EPOCHS = {
        1: 10,
        2: 10,
        3: 15,
        4: 15,
        5: 10
    }

    accumulated_epochs = 0
    for p in range(1, start_phase):
        accumulated_epochs += PHASE_EPOCHS.get(p, 10)
    
    for phase_id in range(start_phase, total_phases + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"PHASE {phase_id} / {total_phases} START")
        logger.info(f"{'='*40}\n")
        
        model.update_phase(phase_id)
        
        epochs_per_phase = PHASE_EPOCHS.get(phase_id, 10)
        
        checkpoint_cb = ModelCheckpoint(
            dirpath=CHECKPOINT_DIR / biome / f"phase_{phase_id}",
            filename=f"{biome}_p{phase_id}_" + "{epoch:02d}_{val_loss:.4f}",
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=True)
        
        trainer = pl.Trainer(
            default_root_dir=LOGS_DIR / biome,
            max_epochs=accumulated_epochs + epochs_per_phase,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=[checkpoint_cb, lr_monitor, early_stop],
            enable_progress_bar=False,
            log_every_n_steps=30,
            gradient_clip_val=0.5,
            precision='16-mixed',
            reload_dataloaders_every_n_epochs=1
        )
        
        ckpt_to_load = None
        if phase_id == start_phase and ckpt_path:
            ckpt_to_load = ckpt_path
            logger.info(f"Resuming from checkpoint: {ckpt_to_load}")
        
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_to_load)
        
        accumulated_epochs += trainer.current_epoch
        accumulated_epochs = trainer.current_epoch 

        logger.info(f"Phase {phase_id} complete. Best loss: {checkpoint_cb.best_model_score}")
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


    logger.info('Training pipeline completed successfully!')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V18.3 Training Pipeline")
    parser.add_argument("biome", type=str, nargs='?', default='amazonia', help="Biome to train")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--limit_fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g. 0.01)")
    parser.add_argument("--chunk_size", type=int, default=350, help="Chunk size in pixels (200=6km, 350=10.5km)")
    parser.add_argument("--test_mode", action='store_true', help="Enable single-tile overhaul test mode")
    parser.add_argument("--start_phase", type=int, default=1, help="Phase to start training from (1-5)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    run_training_pipeline(
        biome=args.biome,
        num_workers=args.workers,
        batch_size_arg=args.batch_size,
        limit_fraction=args.limit_fraction,
        test_mode=args.test_mode,
        chunk_size=args.chunk_size,
        start_phase=args.start_phase,
        ckpt_path=args.ckpt_path
    )
