# test_transformer_block_decoder.py

import torch
from model.transformer_block_decoder import TransformerDecoderBlock

# Example parameters
batch_size = 2
seq_len = 5
embed_dim = 16
num_heads = 4
ff_hidden_dim = 32

# Example dummy input
x = torch.randn(batch_size, seq_len, embed_dim)

# Create an instance of our decoder block
decoder_block = TransformerDecoderBlock(embed_dim, num_heads, ff_hidden_dim)

# Forward pass
out = decoder_block(x)

print("Output shape:", out.shape)  # Expected: [2, 5, 16]
