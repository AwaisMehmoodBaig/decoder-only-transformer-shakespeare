import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadSelfAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
                
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # First sub-layer: Masked self-attention + residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, mask=mask)
        x = x + attn_out

        # Second sub-layer: Feed-forward + residual
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + ff_out

        return x
