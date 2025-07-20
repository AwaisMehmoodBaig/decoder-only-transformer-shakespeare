import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))  # Learnable scale
        self.beta = nn.Parameter(torch.zeros(embed_dim))  # Learnable shift
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        mean = x.mean(dim=-1, keepdim=True)         # Mean along embedding dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Variance along embedding dim
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # Normalize

        return self.gamma * x_norm + self.beta  # Scale and shift
