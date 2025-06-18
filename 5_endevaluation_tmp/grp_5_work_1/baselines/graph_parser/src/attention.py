import torch
import torch.nn.functional as F
from utils import batched_concat_per_row, create_parameter


class Attention:
    @staticmethod
    def edge_factory(dim, attention_type):
        if attention_type == "bilinear":
            return BilinearEdgeAttention(dim)
        elif attention_type == "biaffine":
            return BiaffineEdgeAttention(dim)
        elif attention_type == "affine":
            return AffineEdgeAttention(dim)
        else:
            raise Exception("{attention_type} is not a valid attention type".format(attention_type))

    @staticmethod
    def label_factory(dim, n_labels, attention_type):
        if attention_type == "bilinear":
            return BilinearLabelAttention(dim, n_labels)
        elif attention_type == "biaffine":
            return BiaffineLabelAttention(dim, n_labels)
        elif attention_type == "affine":
            return AffineLabelAttention(dim, n_labels)
        else:
            raise Exception("{attention_type} is not a valid attention type".format(attention_type))

    def get_label_scores(self, head, dep):
        # head, dep: [sequence x batch x mlp]
        raise NotImplementedError()

    def get_edge_scores(self, head, dep):
        # head, dep: [sequence x batch x mlp]
        raise NotImplementedError()


class BilinearEdgeAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_U = create_parameter(dim, dim)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]

        # (batch x seq x seq)
        return torch.einsum("bij,jk,bok->bio", (head, self.edge_U, dep))


class BilinearLabelAttention(torch.nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()
        self.label_U_diag = create_parameter(n_labels, dim)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]

        # (batch x label x seq x seq)
        return torch.einsum("bij,lj,boj->blio", (head, self.label_U_diag, dep))


class BiaffineEdgeAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_U = create_parameter(dim, dim)
        self.edge_W = create_parameter(1, 2 * dim)
        self.edge_b = create_parameter(1)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)
        device = head.device
        
        # Ensure all parameters are on the correct device
        edge_U = self.edge_U.to(device)
        edge_W = self.edge_W.to(device)
        edge_b = self.edge_b.to(device)

        # (batch x seq x seq)
        t1 = torch.einsum("bij,jk,bok->bio", (head, edge_U, dep))

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (1 x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x 1 x seq*seq)
        t2 = edge_W @ concated.transpose(1, 2)

        # (batch x 1 x seq*seq) => (batch x seq x seq)
        t2 = t2.view(batch_size, sequence_size, sequence_size)

        return t1 + t2 + edge_b


class BiaffineLabelAttention(torch.nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()

        self.label_U_diag = create_parameter(n_labels, dim)
        self.label_W = create_parameter(n_labels, 2 * dim)
        self.label_b = create_parameter(n_labels)
        self.n_labels = n_labels

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)
        device = head.device
        
        # Ensure all parameters are on the correct device
        label_U_diag = self.label_U_diag.to(device)
        label_W = self.label_W.to(device)
        label_b = self.label_b.to(device)

        # (batch x label x seq x seq)
        t1 = torch.einsum("bij,lj,boj->blio", (head, label_U_diag, dep))

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (labels x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x labels x seq*seq)
        t2 = label_W @ concated.transpose(1, 2)

        # (batch x labels x seq*seq) => (batch x labels x seq x seq)
        t2 = t2.view(batch_size, self.n_labels, sequence_size, sequence_size)

        return t1 + t2 + label_b[None, :, None, None]


class AffineLabelAttention(torch.nn.Module):
    def __init__(self, dim, n_labels):
        super().__init__()

        self.label_W = create_parameter(n_labels, 2 * dim)
        self.label_b = create_parameter(n_labels)
        self.n_labels = n_labels

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)
        device = head.device
        
        # Ensure parameters are on the correct device
        label_W = self.label_W.to(device)
        label_b = self.label_b.to(device)

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (labels x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x labels x seq*seq)
        t2 = label_W @ concated.transpose(1, 2)

        # (batch x labels x seq*seq) => (batch x labels x seq x seq)
        t2 = t2.view(batch_size, self.n_labels, sequence_size, sequence_size)

        return t2 + label_b[None, :, None, None]


class AffineEdgeAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_W = create_parameter(1, 2 * dim)
        self.edge_b = create_parameter(1)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        batch_size = head.size(0)
        sequence_size = head.size(1)
        device = head.device
        
        # Ensure parameters are on the correct device
        edge_W = self.edge_W.to(device)
        edge_b = self.edge_b.to(device)

        # (batch x seq*seq x 2mlp)
        concated = batched_concat_per_row(head, dep)

        # (1 x 2mlp) @ (batch x 2mlp x seq*seq) => (batch x 1 x seq*seq)
        t2 = edge_W @ concated.transpose(1, 2)

        # (batch x 1 x seq*seq) => (batch x seq x seq)
        t2 = t2.view(batch_size, sequence_size, sequence_size)

        return t2 + edge_b



class DotProductAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dk = dim ** 0.5#torch.sqrt(dk)

    def forward(self, attention_matrix, output):
        # TODO really dim=1?
        attention_matrix = attention_matrix
        am = F.softmax(attention_matrix.transpose(-2,-1) * self.dk, dim=1) @ output
        return am


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention implementation for enhanced representation learning"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, query, key, value, mask=None):
        device = query.device
        batch_size = query.size(0)
        
        # Make sure all modules are on the same device as input
        self.q_linear = self.q_linear.to(device)
        self.k_linear = self.k_linear.to(device)
        self.v_linear = self.v_linear.to(device)
        self.output_linear = self.output_linear.to(device)
        
        # Linear projections and split into heads
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.to(device)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original size
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim)
        
        output = self.output_linear(attn_output)
        
        return output, attn_weights

class MultiAttentionGATLayer(torch.nn.Module):
    """Enhanced GAT layer with multiple attention mechanisms"""
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1, alpha=0.2, 
                 concat=True, residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual
        
        # If we're concatenating heads, we need to divide the output dimension
        self.head_dim = out_dim // num_heads if concat else out_dim
        
        # Primary attention parameters
        self.W = torch.nn.Parameter(torch.zeros(num_heads, in_dim, self.head_dim))
        self.a1 = torch.nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim, 1))
        
        # Secondary attention parameters (biaffine)
        self.U = torch.nn.Parameter(torch.zeros(num_heads, self.head_dim, self.head_dim))
        self.a2 = torch.nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        self.bias = torch.nn.Parameter(torch.zeros(num_heads, 1))
        
        # Meta-attention to learn attention mechanism weights
        self.meta_attention = torch.nn.Parameter(torch.ones(2))
        
        # Transformer-like attention
        self.multi_head_attn = MultiHeadAttention(self.head_dim, 4, dropout)
        
        # Regularization
        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        
        # LayerNorm for stability
        self.layer_norm1 = torch.nn.LayerNorm(out_dim if concat else self.head_dim)
        self.layer_norm2 = torch.nn.LayerNorm(out_dim if concat else self.head_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize learnable parameters using Glorot (Xavier) initialization"""
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.a1)
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.a2.view(self.num_heads, -1, 1))
        
    def forward(self, x, adj):
        """
        x: node features (batch_size, num_nodes, in_dim)
        adj: adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        device = x.device  # Get the device of the input
        batch_size, num_nodes = x.size(0), x.size(1)
        
        # Make sure all parameters are on the same device as input
        W = self.W.to(device)
        a1 = self.a1.to(device)
        U = self.U.to(device)
        a2 = self.a2.to(device)
        bias = self.bias.to(device)
        meta_attention = self.meta_attention.to(device)
        
        # Apply linear transformation to each head
        Wh = torch.einsum('hid,bni->bnhd', (W, x))  # (batch, nodes, heads, head_dim)
        
        # Self-attention (GAT-style)
        # Repeat to create pairs for attention
        a_input1 = torch.repeat_interleave(Wh, num_nodes, dim=1)  # (batch, nodes*nodes, heads, head_dim)
        a_input2 = Wh.repeat(1, num_nodes, 1, 1)  # (batch, nodes*nodes, heads, head_dim)
        a_input = torch.cat([a_input1, a_input2], dim=-1)  # (batch, nodes*nodes, heads, 2*head_dim)
        
        # Primary attention (GAT)
        e1 = torch.einsum('bnhf,hfi->bnhi', (a_input, a1))
        e1 = e1.squeeze(-1).view(batch_size, num_nodes, num_nodes, self.num_heads)
        e1 = self.leakyrelu(e1)
        
        # Secondary attention (Biaffine)
        e2_1 = torch.einsum('bnhd,hdj->bnhj', (Wh, U))  # (batch, nodes, heads, head_dim)
        e2 = torch.einsum('bnhd,bmhd->bnmh', (e2_1, Wh))  # (batch, nodes, nodes, heads)
        
        # Add biaffine bias
        Wh_reshaped = Wh.view(batch_size, num_nodes, self.num_heads * self.head_dim)
        a2_reshaped = a2.view(self.num_heads, -1)
        e2_bias = torch.einsum('bni,hj->bnih', (Wh_reshaped, a2_reshaped))
        e2_bias = e2_bias.view(batch_size, num_nodes, num_nodes, self.num_heads)
        e2 = e2 + e2_bias + bias  # (batch, nodes, nodes, heads)
        
        # Normalize meta-attention weights using softmax
        meta_weights = F.softmax(meta_attention, dim=0)
        
        # Combine attention mechanisms
        e = meta_weights[0] * e1 + meta_weights[1] * e2
        
        # Apply adjacency matrix mask
        adj = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        masked_e = e.masked_fill(adj == 0, -9e15)
        
        # Apply attention scores
        attention = F.softmax(masked_e, dim=2)  # (batch, nodes, nodes, heads)
        attention = self.dropout(attention)
        
        # Apply attention weights to node features
        h_prime = torch.einsum('bnmh,bmhd->bnhd', (attention, Wh))  # (batch, nodes, heads, head_dim)
        
        # Concat or average heads
        if self.concat:
            h_prime = h_prime.view(batch_size, num_nodes, -1)  # (batch, nodes, heads*head_dim)
        else:
            h_prime = h_prime.mean(dim=2)  # (batch, nodes, head_dim)
            
        # Apply transformer-like self-attention for global context
        h_prime_transformed, _ = self.multi_head_attn(h_prime, h_prime, h_prime)
        
        # Apply residual connection, normalization, and optional residual
        h_prime = self.layer_norm1(h_prime)
        h_prime_transformed = self.layer_norm2(h_prime_transformed)
        
        # Final output with residual connection
        if self.residual and x.size(-1) == h_prime.size(-1):
            output = x + 0.5 * h_prime + 0.5 * h_prime_transformed
        else:
            output = 0.5 * h_prime + 0.5 * h_prime_transformed
            
        return output


class MultiAttentionGAT(torch.nn.Module):
    """Full GAT model with multiple attention mechanisms and residual connections"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=8, 
                 dropout=0.1, alpha=0.2, residual=True):
        super().__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(MultiAttentionGATLayer(
            in_dim, hidden_dim, num_heads, dropout, alpha, True, False))
        
        # Hidden layers
        for i in range(1, num_layers - 1):
            self.convs.append(MultiAttentionGATLayer(
                hidden_dim, hidden_dim, num_heads, dropout, alpha, True, residual))
            
        # Output layer
        self.convs.append(MultiAttentionGATLayer(
            hidden_dim, out_dim, num_heads, dropout, alpha, False, residual))
            
        # Regularization
        self.dropout = torch.nn.Dropout(dropout)
        
        # Global pooling
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, adj):
        """
        x: node features (batch_size, num_nodes, in_dim)
        adj: adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # Initial dropout
        x = self.dropout(x)
        
        # Apply GAT layers with residual connections
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)
            if i != len(self.convs) - 1:  # No dropout after last layer
                x = self.dropout(x)
                
        return x

