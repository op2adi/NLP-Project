# gat_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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
        x_flat = x.view(batch_size * N, -1)

        # apply GATConv
        h = F.elu(self.gat1(x_flat, edge_index))
        h = self.dropout(h)
        h = self.gat2(h, edge_index)

        # back to batch x N x out_dim
        return h.view(batch_size, N, -1)
