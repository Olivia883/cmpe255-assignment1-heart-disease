from pathlib import Path
import json
import math
import torch
import torch.nn as nn

TEXT = Path("data.txt").read_text(encoding="utf-8")
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

batch_size = 16
block_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(source):
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layer)
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        t = x.size(1)
        positions = torch.arange(t, device=x.device)
        h = self.token(x) + self.pos(positions)[None, :, :]

        mask = torch.triu(
            torch.ones(t, t, device=x.device),
            diagonal=1
        ).bool()

        h = self.encoder(h, mask=mask)
        logits = self.head(self.norm(h))

        loss = None
        if y is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
        return logits, loss

model = TinyTransformer(len(chars)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

history = []

for step in range(300):
    model.train()
    x, y = get_batch(train_data)
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        history.append({"step": step, "train_loss": float(loss)})
        print(step, float(loss))

torch.save(
    {
        "model": model.state_dict(),
        "stoi": stoi,
        "itos": itos
    },
    "model.pt"
)

Path("metrics.json").write_text(
    json.dumps(history, indent=2)
)

print("Saved model.pt and metrics.json")