from pathlib import Path
import torch
import torch.nn as nn

checkpoint = torch.load("model.pt", map_location="cpu")
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 32

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

    def forward(self, x):
        t = x.size(1)
        positions = torch.arange(t, device=x.device)
        h = self.token(x) + self.pos(positions)[None, :, :]
        mask = torch.triu(
            torch.ones(t, t, device=x.device),
            diagonal=1
        ).bool()
        h = self.encoder(h, mask=mask)
        return self.head(self.norm(h))

model = TinyTransformer(len(stoi)).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

def sample(prompt, steps=120):
    ids = [stoi.get(c, 0) for c in prompt]
    x = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)

    out = list(ids)
    for _ in range(steps):
        logits = model(x)
        next_id = torch.multinomial(
            torch.softmax(logits[:, -1, :], dim=-1),
            1
        ).item()
        out.append(next_id)
        x = torch.tensor(
            [out[-block_size:]],
            dtype=torch.long,
            device=device
        )
    return "".join(itos[i] for i in out)

print("Nano LLM chatbot. Type 'quit' to exit.")
while True:
    prompt = input("> ")
    if prompt.lower() == "quit":
        break
    print(sample(prompt))
