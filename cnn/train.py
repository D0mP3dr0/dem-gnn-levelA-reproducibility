INTERNAL_DATA_ROOT = __import__("os").environ.get("DEMGNN_DATA_ROOT", ".")
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import torch

import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.models.unet_model import CNNTrainer
from src.data.dataset import CNNDataset
from src.utils.console_logger import RichMetricCallback

def main(args):
    pl.seed_everything(42)
    
    suffix = "lidar" if args.use_lidar_input else "sr"
    print(f"Loading data ({suffix}) for: {args.biome}")
    train_ds = CNNDataset(args.data_dir, args.biome, mode='train', augment=True, patch_size=args.patch_size, use_lidar_input=args.use_lidar_input)
    val_ds = CNNDataset(args.data_dir, args.biome, mode='val', augment=False, patch_size=args.patch_size, use_lidar_input=args.use_lidar_input)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = CNNTrainer(biome=args.biome, lr=args.lr, use_lidar_input=args.use_lidar_input)
    
    checkpoint_dir = f"{args.checkpoint_dir}/{args.biome}_{suffix}"
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_mae:.4f}",
        monitor="val_mae",
        mode="min",
        save_top_k=1
    )
    
    early_stop_cb = EarlyStopping(
        monitor="val_mae",
        patience=10,
        mode="min",
        verbose=True
    )
    
    console_cb = RichMetricCallback()
    
    logger = CSVLogger(args.results_dir, name=args.biome)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb, console_cb],
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    print("Iniciando treinamento...")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Best model saved to: {checkpoint_cb.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--biome", type=str, required=True, help="amazonia, cerrado, pantanal, mata_atlantica")
    parser.add_argument("--use_lidar_input", action="store_true", help='If True, uses 19 channels (Fusion). Otherwise, 17 (Pure SR).')
    
    parser.add_argument("--data_dir", type=str, default=f"{INTERNAL_DATA_ROOT}/RESU-NET/data_cnn")
    parser.add_argument("--checkpoint_dir", type=str, default=f"{INTERNAL_DATA_ROOT}/RESU-NET/ResU-Net_RUN_COLAB/checkpoints")
    parser.add_argument("--results_dir", type=str, default=f"{INTERNAL_DATA_ROOT}/RESU-NET/ResU-Net_RUN_COLAB/results")
    
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patch_size", type=int, default=384, help="Tamanho do recorte (default 384)")
    
    args = parser.parse_args()
    main(args)
