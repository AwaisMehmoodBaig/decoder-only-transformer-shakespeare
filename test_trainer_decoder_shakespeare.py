import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_block_decoder import TransformerDecoderBlock

# ✅ Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Model parameters 
vocab_size = tokenizer["vocab_size"]
embed_dim = 128
num_heads = 4
ff_hidden_dim = 256
num_layers = 4
max_seq_len = 1000  

# ✅ Define the model
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed[:, :T, :]
        x = tok_emb + pos_emb

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.fc_out(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None, top_p=None, do_sample=True):
        for _ in range(max_length - input_ids.shape[1]):
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)

            if torch.any(torch.isnan(probs)) or torch.any(probs < 0) or torch.any(torch.isinf(probs)):
                print("⚠️ Bad probs detected, regenerating with safe fallback...")
                probs = F.softmax(torch.clamp(logits, -10, 10), dim=-1)

            next_token = (
                torch.multinomial(probs, num_samples=1)
                if do_sample else
                torch.argmax(probs, dim=-1, keepdim=True)
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ✅ Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DecoderOnlyTransformer(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len).to(device)
model.load_state_dict(torch.load("decoder_model.pth", map_location=device))
model.eval()

# ✅ Prompt
prompt = "Once in a land far far away"
input_ids = torch.tensor([[tokenizer["stoi"][c] for c in prompt]], dtype=torch.long).to(device)

# ✅ Generate
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

# ✅ Decode output
decoded = ''.join([tokenizer["itos"][i] for i in output_ids[0].tolist()])
print("=" * 50)
print(f"Prompt: {prompt}")
print("-" * 50)
print(f"Generated: {decoded}")
print("=" * 50)
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")