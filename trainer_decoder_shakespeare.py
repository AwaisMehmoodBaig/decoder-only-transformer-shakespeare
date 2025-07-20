import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model.transformer_block_decoder import TransformerDecoderBlock

# 💡 Parameters
embed_dim = 128
num_heads = 4
ff_hidden_dim = 256
num_layers = 4
seq_len = 100
batch_size = 16
num_epochs = 5
lr = 1e-4

# 📄 Load text
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Split into train (90%) and val (10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 📄 Dataset
class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.seq_len+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

train_dataset = CharDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 📄 Model
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # Up to 1000 positions

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

        # Generate causal mask
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0).to(x.device)

        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DecoderOnlyTransformer(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 💥 Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        B, T, C = logits.shape

        loss = criterion(logits.view(B*T, C), batch_y.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ✅ Print batch progress every 10000 batches
        if (i + 1) % 10000 == 0:
            print(f"Batch {i+1}/{len(train_loader)} — Batch Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

# ✅ Save model
torch.save(model.state_dict(), "decoder_model.pth")
print("✅ Model saved to decoder_model.pth")
print("✅ Training finished!")
