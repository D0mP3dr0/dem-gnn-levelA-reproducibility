import torch
import pytorch_lightning as pl
import logging
import gc
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Optional

from data.dataset_v18 import SpatialHeteroGraphDataset
from config import TRAINING_CONFIG

logger = logging.getLogger(__name__)

class HierarchicalDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 batch_size: int = 1,
                 chunk_size: int = 512,
                 num_workers: int = 0,
                 val_split: float = 0.2,
                 biome: str = 'default',
                 test_mode: bool = False,
                 limit_fraction: float = 1.0):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.biome = biome
        self.test_mode = test_mode
        self.limit_fraction = limit_fraction
        
        self.data: Optional[HeteroData] = None
        self.train_dataset: Optional[SpatialHeteroGraphDataset] = None
        self.val_dataset: Optional[SpatialHeteroGraphDataset] = None
        self.test_dataset: Optional[SpatialHeteroGraphDataset] = None
        
        logger.info(f"HierarchicalDataModule has been initialized ({biome})")
        logger.info(f"   • Path: {dataset_path}")
        logger.info(f"   - Chunk size: {chunk_size}")
        if test_mode:
            logger.info('   TEST MODE ENABLED: Restricting to Single Middle Tile (Overfitting Check)')

    def prepare_data(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.dataset_path}")

    def setup(self, stage: Optional[str] = None):
        if self.data is not None:
            return

        logger.info('Loading HeteroGraph into RAM...')
        self.data = torch.load(self.dataset_path, map_location='cpu', weights_only=False)
        
        full_dataset = SpatialHeteroGraphDataset(
            self.data, 
            chunk_size=self.chunk_size, 
            overlap=32,
            biome=self.biome
        )
        
        if self.test_mode:
            mid_idx = len(full_dataset) // 2
            full_dataset.chunks = [full_dataset.chunks[mid_idx]]
            logger.info(f"   TEST MODE: Subsetting to chunk index {mid_idx} (Coords: {full_dataset.chunks[0]})")
            
            self.train_dataset = full_dataset
            self.val_dataset = full_dataset
            return
        
        if self.limit_fraction < 1.0:
            total_chunks = len(full_dataset)
            keep_count = max(1, int(total_chunks * self.limit_fraction))
            start_idx = (total_chunks - keep_count) // 2
            full_dataset.chunks = full_dataset.chunks[start_idx : start_idx + keep_count]
            logger.info(f"   LIMIT: Using {keep_count}/{total_chunks} chunks ({self.limit_fraction:.4%})")
        
        total_chunks = len(full_dataset)
        val_size = max(1, int(total_chunks * self.val_split))
        train_size = total_chunks - val_size
        
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        logger.info(f"Split created: {train_size} train chunks / {val_size} val chunks")

        self.test_dataset = full_dataset

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0 if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def teardown(self, stage: Optional[str] = None):
        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        gc.collect()
