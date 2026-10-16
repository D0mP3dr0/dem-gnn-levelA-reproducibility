import torch
import torch.utils.data
import logging
from torch_geometric.data import HeteroData
from typing import Optional, List, Dict
from config import NORMALIZATION_STATS
from .transforms_v18 import HeteroComposer

logger = logging.getLogger(__name__)

class SpatialHeteroGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data: HeteroData, chunk_size: int = 350, overlap: int = 32, border_crop: int = 40, biome: str = 'default'):
        logger.info(f"Applying V18.3 transforms for biome: {biome}...")
        
        stats = NORMALIZATION_STATS.get(biome, NORMALIZATION_STATS['default'])
        
        if 'dem' in data.node_types:
             elev_mean_check = data['dem'].x[:, 0].float().mean().item()
             if abs(elev_mean_check) > 5.0:
                 logger.info(f"Raw data detected (mean elev: {elev_mean_check:.2f}). Applying normalization...")
                 transforms = HeteroComposer.get_transforms(stats)
                 self.data = transforms(data)
             else:
                 self.data = data
                 logger.info(f"Data appears pre-normalized (mean elev: {elev_mean_check:.2f}). Skipping.")
        else:
             self.data = data
             logger.warning('No DEM nodes found. Normalization status unknown.')
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.border_crop = border_crop
        
        grid_shape = data['dem'].grid_shape if hasattr(data['dem'], 'grid_shape') else None
        
        if grid_shape is not None:
            if torch.is_tensor(grid_shape):
                shape_list = grid_shape.tolist()
                if len(shape_list) == 2:
                    self.H_dem, self.W_dem = shape_list
                else:
                    import math
                    num_nodes = data['dem'].num_nodes
                    side = int(math.sqrt(num_nodes))
                    self.H_dem, self.W_dem = side, side
            elif hasattr(grid_shape, '__len__') and len(grid_shape) == 2:
                self.H_dem, self.W_dem = grid_shape
            else:
                import math
                num_nodes = data['dem'].num_nodes
                side = int(math.sqrt(num_nodes))
                self.H_dem, self.W_dem = side, side
        else:
            import math
            num_nodes = data['dem'].num_nodes
            side = int(math.sqrt(num_nodes))
            self.H_dem, self.W_dem = side, side

        expected_nodes = self.H_dem * self.W_dem
        actual_nodes = data['dem'].num_nodes
        
        if expected_nodes != actual_nodes:
            logger.warning(f"Dimension mismatch detected! Meta: {self.H_dem}x{self.W_dem} ({expected_nodes}), Actual: {actual_nodes}")
            
            if self.W_dem > 0 and actual_nodes % self.W_dem == 0:
                new_H = actual_nodes // self.W_dem
                logger.info(f"   Adjusting grid height: {self.H_dem} -> {new_H} (preserving width {self.W_dem})")
                self.H_dem = new_H
            else:
                import math
                side = int(math.sqrt(actual_nodes))
                logger.warning(f"   Resetting to square grid: {side}x{side}")
                self.H_dem, self.W_dem = side, side
        
        logger.info(f"Dataset grid: {self.H_dem}x{self.W_dem} (DEM pixels)")

        
        self.chunks = []
        stride = chunk_size - overlap
        
        for r in range(0, self.H_dem, stride):
            for c in range(0, self.W_dem, stride):
                r_end = min(r + chunk_size, self.H_dem)
                c_end = min(c + chunk_size, self.W_dem)
                
                r_valid_start = r + border_crop
                c_valid_start = c + border_crop
                r_valid_end = r_end - border_crop
                c_valid_end = c_end - border_crop
                
                if (r_valid_end - r_valid_start) > 50 and (c_valid_end - c_valid_start) > 50:
                    self.chunks.append({
                        'full': (r, c, r_end, c_end),
                        'valid': (r_valid_start, c_valid_start, r_valid_end, c_valid_end)
                    })
                
        logger.info(f"Generated {len(self.chunks)} spatial chunks (Stride: {stride}, Border Crop: {border_crop})")

        if 'lidar' in data.node_types and data['lidar'].num_nodes > 0:
            logger.info("Indexing LiDAR nodes by parent_idx for fast slicing...")
            self.lidar_sorted_order = torch.argsort(data['lidar'].parent_idx)
            self.lidar_sorted_parents = data['lidar'].parent_idx[self.lidar_sorted_order]
        else:
            self.lidar_sorted_order = None
            self.lidar_sorted_parents = None

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]
        r0, c0, r1, c1 = chunk_info['full']
        r_valid_start, c_valid_start, r_valid_end, c_valid_end = chunk_info['valid']
        
        dem_indices_list = []
        for r in range(r0, r1):
            start = r * self.W_dem + c0
            end = start + (c1 - c0)
            dem_indices_list.append(torch.arange(start, end, device=self.data['dem'].x.device))
        
        dem_nodes = torch.cat(dem_indices_list)
        
        r_sent, c_sent = r0 * 3, c0 * 3
        h_sent = (r1 - r0) * 3
        w_sent = (c1 - c0) * 3
        W_S = self.W_dem * 3
        
        sent_indices_list = []
        for r in range(r_sent, r_sent + h_sent):
            start = r * W_S + c_sent
            end = start + w_sent
            sent_indices_list.append(torch.arange(start, end, device=self.data['dem'].x.device))
        sent_nodes = torch.cat(sent_indices_list)
        
        node_idx_dict = {
            'dem': dem_nodes,
            'sentinel': sent_nodes
        }
        
        if self.lidar_sorted_order is not None:
            min_parent = dem_nodes.min().item()
            max_parent = dem_nodes.max().item()
            
            l_start = torch.searchsorted(self.lidar_sorted_parents, min_parent, side='left')
            l_end = torch.searchsorted(self.lidar_sorted_parents, max_parent, side='right')
            
            candidate_indices = self.lidar_sorted_order[l_start:l_end]
            
            candidate_parents = self.data['lidar'].parent_idx[candidate_indices]
            
            mask = torch.isin(candidate_parents, dem_nodes)
            lidar_nodes = candidate_indices[mask]
            
            node_idx_dict['lidar'] = lidar_nodes

        batch = self.data.subgraph(node_idx_dict)
        
        node_types_to_remove = []
        for node_type in batch.node_types:
            if batch[node_type].num_nodes == 0:
                node_types_to_remove.append(node_type)
        
        for node_type in node_types_to_remove:
            del batch[node_type]
        
        edge_types_to_remove = []
        for edge_type in batch.edge_types:
            src_type, _, dst_type = edge_type
            if src_type in node_types_to_remove or dst_type in node_types_to_remove:
                edge_types_to_remove.append(edge_type)
                continue
            edge_index = batch[edge_type].edge_index
            if edge_index.numel() == 0:
                edge_types_to_remove.append(edge_type)
        
        for edge_type in edge_types_to_remove:
            del batch[edge_type]
        
        batch.chunk_info = torch.tensor([r0, c0, r1, c1])
        batch.valid_region = torch.tensor([r_valid_start, c_valid_start, r_valid_end, c_valid_end])
        
        return batch


