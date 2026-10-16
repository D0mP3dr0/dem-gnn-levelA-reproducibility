import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
try:
    from torch_scatter import scatter_add
except ImportError:
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim_size is None:
            dim_size = index.max().item() + 1 if len(index) > 0 else 0
        
        output = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        for i in range(len(index)):
            output[index[i]] += src[i]
        return output
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EdgeTypeConv(MessagePassing):
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_edge_types: int, edge_dim: int = 0,
                 aggr: str = 'add', bias: bool = True):
        super(EdgeTypeConv, self).__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.edge_dim = edge_dim
        
        self.edge_type_linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=bias)
            for _ in range(num_edge_types)
        ])
        
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        else:
            self.edge_encoder = None
        
        self.edge_type_attention = nn.Parameter(torch.randn(num_edge_types, 1))
        
        self.output_proj = nn.Linear(out_channels, out_channels, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for linear in self.edge_type_linears:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
        if self.edge_encoder is not None:
            for layer in self.edge_encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.edge_type_attention)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        self.edge_type = edge_type
        self.edge_attr = edge_attr
        
        out = self.propagate(edge_index, x=x, size=None)
        
        out = self.output_proj(out)
        
        return out
    
    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, 
                edge_index_i: torch.Tensor) -> torch.Tensor:
        edge_indices = torch.arange(x_j.size(0), device=x_j.device)
        
        messages = torch.zeros_like(x_j)
        
        for edge_type_idx in range(self.num_edge_types):
            type_mask = (self.edge_type == edge_type_idx)
            if not type_mask.any().item():
                continue
            
            x_j_type = x_j[type_mask]
            
            transformed = self.edge_type_linears[edge_type_idx](x_j_type)
            
            if self.edge_encoder is not None and self.edge_attr is not None:
                edge_attr_type = self.edge_attr[type_mask]
                edge_features = self.edge_encoder(edge_attr_type)
                transformed = transformed + edge_features
            
            attention_weight = torch.sigmoid(self.edge_type_attention[edge_type_idx])
            transformed = transformed * attention_weight
            
            messages[type_mask] = transformed
        
        return messages
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return aggr_out

class EdgeTypeAttention(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 8, dropout: float = 0.1,
                 edge_dim: int = 0, bias: bool = True):
        super(EdgeTypeAttention, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        assert out_channels % heads == 0
        self.head_dim = out_channels // heads
        
        self.q_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.k_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.v_proj = nn.Linear(in_channels, out_channels, bias=bias)
        
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.edge_proj = None
        
        self.out_proj = nn.Linear(out_channels, out_channels, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.attention_weights = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.size(0)
        
        Q = self.q_proj(x).view(N, self.heads, self.head_dim)
        K = self.k_proj(x).view(N, self.heads, self.head_dim)
        V = self.v_proj(x).view(N, self.heads, self.head_dim)
        
        out = self._compute_attention(Q, K, V, edge_index, edge_attr)
        
        out = out.view(N, self.out_channels)
        out = self.out_proj(out)
        
        return out
    
    def _compute_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          edge_index: torch.Tensor, 
                          edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        E = edge_index.size(1)
        heads = self.heads
        head_dim = self.head_dim
        
        row, col = edge_index[0], edge_index[1]
        
        q_edges = Q[col]
        k_edges = K[row]
        
        scores = (q_edges * k_edges).sum(dim=-1) / (head_dim ** 0.5)
        
        if edge_attr is not None and self.edge_proj is not None:
            edge_scores = self.edge_proj(edge_attr)
            scores = scores + edge_scores
        
        attention = softmax(scores, col, num_nodes=Q.size(0))
        
        self.attention_weights = attention.detach()
        
        attention = self.dropout_layer(attention)
        
        v_edges = V[row]
        weighted_values = attention.unsqueeze(-1) * v_edges
        
        out = scatter_add(weighted_values, col, dim=0, dim_size=Q.size(0))
        
        return out

class TerrainSpecificLayer(nn.Module):
    
    def __init__(self, hidden_dim: int, num_edge_types: int = 3, 
                 edge_dim: int = 4, dropout: float = 0.1):
        super(TerrainSpecificLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        
        self.edge_conv = EdgeTypeConv(
            hidden_dim, hidden_dim, num_edge_types, edge_dim
        )
        
        self.attention = EdgeTypeAttention(
            hidden_dim, hidden_dim, heads=8, dropout=dropout, edge_dim=edge_dim
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.edge_conv(x, edge_index, edge_type, edge_attr)
        x = self.dropout(x)
        x = self.layer_norm1(x + residual)
        
        residual = x
        x = self.attention(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.layer_norm2(x + residual)
        
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        
        return x

class EdgeTypeEncoder(nn.Module):
    
    def __init__(self, edge_dim: int, hidden_dim: int, num_edge_types: int = 3):
        super(EdgeTypeEncoder, self).__init__()
        
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        
        if edge_dim > 0:
            self.edge_attr_proj = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.edge_attr_proj = None
        
        self.combination = nn.Sequential(
            nn.Linear(hidden_dim * (2 if edge_dim > 0 else 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, edge_type: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        type_emb = self.edge_type_embedding(edge_type)
        
        if edge_attr is not None and self.edge_attr_proj is not None:
            attr_emb = self.edge_attr_proj(edge_attr)
            combined = torch.cat([type_emb, attr_emb], dim=-1)
        else:
            combined = type_emb
        
        edge_emb = self.combination(combined)
        
        return edge_emb 
