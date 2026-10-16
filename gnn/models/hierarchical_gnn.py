import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, HeteroConv, Linear, BatchNorm
from torch.utils.checkpoint import checkpoint
import logging

logger = logging.getLogger(__name__)

class HierarchicalTerrainGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=256, num_layers=4, heads=2, edge_dim=2, dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.metadata = metadata
        
        self.dem_encoder = nn.Sequential(
            Linear(17, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.sent_encoder = nn.Sequential(
            Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lidar_encoder = nn.Sequential(
            Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {
                ('dem', 'adjacent_to', 'dem'): GATv2Conv(
                    hidden_dim, hidden_dim, heads=heads, 
                    concat=False, edge_dim=edge_dim, add_self_loops=False
                ),
                
                ('sentinel', 'belongs_to', 'dem'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1,
                    concat=False, edge_dim=3, add_self_loops=False
                ),
                
                ('lidar', 'belongs_to', 'dem'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1,
                    concat=False, edge_dim=2, add_self_loops=False
                ),
                
                ('lidar', 'near_to', 'dem'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1, 
                    concat=False, edge_dim=2, add_self_loops=False
                )
            }
            
            if ('lidar', 'near_to', 'lidar') in list(zip(*metadata[1])):
                conv_dict[('lidar', 'near_to', 'lidar')] = GATv2Conv(
                    hidden_dim, hidden_dim, heads=1,
                    concat=False, edge_dim=4, add_self_loops=False
                )
            
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                'dem': nn.LayerNorm(hidden_dim),
            }))

        target_names = [
            'elevation', 'slope', 'aspect_cos', 'aspect_sin', 
            'curvature', 'tpi', 'tri', 'roughness', 'flow_accum',
            'ndvi', 'ndwi', 'bsi', 'shadow'
        ]
        
        heads_dict = {}
        for name in target_names:
            heads_dict[name] = nn.Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                Linear(hidden_dim // 2, 1)
            )
            
        heads_dict['canopy'] = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2), 
            nn.GELU(), 
            Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.heads = nn.ModuleDict(heads_dict)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = x_dict.copy()
        x_dict['dem'] = self.dem_encoder(x_dict['dem'])
        x_dict['sentinel'] = self.sent_encoder(x_dict['sentinel'])
        if 'lidar' in x_dict:
            x_dict['lidar'] = self.lidar_encoder(x_dict['lidar'])
            
        for conv, norm_dict in zip(self.convs, self.norms):
            out_dict = checkpoint(conv, x_dict, edge_index_dict, edge_attr_dict, use_reentrant=False)
            
            for key in out_dict:
                if key in x_dict:
                    out = out_dict[key] + x_dict[key] 
                else:
                    out = out_dict[key]
                    
                if key in norm_dict:
                    out = norm_dict[key](out)
                out = F.gelu(out)
                
                x_dict[key] = out
                
        dem_h = x_dict['dem']
        preds = {}
        for name, head in self.heads.items():
            preds[name] = head(dem_h)
            
        return preds
        
    def update_phase(self, phase):
        logger.info(f"Model updated to PHASE {phase}")
