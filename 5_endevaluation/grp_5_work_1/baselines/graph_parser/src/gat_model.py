# gat_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from attention import MultiAttentionGAT, MultiAttentionGATLayer

class GATBridge(nn.Module):
    """
    A simple 2‑layer GAT to replace the GCN bridge.
    Expects:
      - x:     [batch, seq_len, in_dim]
      - adj:   [batch, seq_len, seq_len]  (0/1 mask)
    Outputs:
      - h_out: [batch, seq_len, out_dim]
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # first layer: multi‑head, concatenated
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads,
                             dropout=dropout, concat=True)
        # second layer: single head, averaged
        self.gat2 = GATConv(hidden_dim * heads, out_dim,
                             heads=1, concat=False,
                             dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        # adj: batch x N x N mask → we need edge_index per graph
        # flatten batch into a single big graph with offsets
        batch_size, N, _ = adj.size()
        device = x.device

        # build a big graph by shifting node indices per batch
        edge_indices = []
        for b in range(batch_size):
            mask = adj[b].nonzero(as_tuple=False)  # [E_b, 2]
            # shift node IDs by b * N
            edge_indices.append(mask + b * N)
        edge_index = torch.cat(edge_indices, dim=0).t().contiguous()
        # reshape x to (batch*N) x in_dim
        x_flat = x.reshape(batch_size * N, -1)

        # apply GATConv
        h = F.elu(self.gat1(x_flat, edge_index))
        h = self.dropout(h)
        h = self.gat2(h, edge_index)

        # back to batch x N x out_dim
        return h.reshape(batch_size, N, -1)


class MultiAttentionGATBridge(nn.Module):
    """
    Enhanced multi-head GAT with multiple attention mechanisms and transformer-like features
    designed to significantly improve F1 scores on structured sentiment tasks.
    
    Expects:
      - x:     [batch, seq_len, in_dim]
      - adj:   [batch, seq_len, seq_len]  (0/1 mask)
    Outputs:
      - h_out: [batch, seq_len, out_dim]
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3,
                 heads: int = 8, dropout: float = 0.15, alpha: float = 0.2, 
                 residual: bool = True):
        super().__init__()
        
        # Use our custom MultiAttentionGAT 
        self.multi_gat = MultiAttentionGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            num_heads=heads,
            dropout=dropout,
            alpha=alpha,
            residual=residual
        )
        
        # Extra layers for better feature extraction
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward output network for refinement
        self.ff_network = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, adj, x):
        # Ensure tensors are on the same device
        device = x.device
        adj = adj.to(device)
        self.multi_gat = self.multi_gat.to(device)
        self.layer_norm = self.layer_norm.to(device)
        self.ff_network = self.ff_network.to(device)
        
        # Apply the multi-attention GAT
        h = self.multi_gat(x, adj)
        
        # Apply layer normalization
        h = self.layer_norm(h)
        
        # Apply feed-forward refinement with residual connection
        h_refined = self.ff_network(h)
        h_out = h + h_refined
        
        return h_out
