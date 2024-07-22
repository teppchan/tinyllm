import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor(
    [[2, 10, 20, 100, 512, 3], [2, 10, 20, 100, 512, 3], [2, 10, 20, 100, 512, 3]],
    dtype=torch.long,
).to(device)

embedding_size = 768
num_heads = 12

gpt = GPT(
    50257,
    embedding_size,
    embedding_size * 4,
    num_heads,
    0.1,
    batch_first=True,
    T=1024,
    N=12,
).to(device)
print(gpt)

print(f"params# = {gpt.calculate_params():,}")

padding_mask, mask = gpt.create_mask(x[0:2], 0, device)
loss, pred = gpt(x[0:2], x[1:3], padding_mask, mask)

if loss is not None:
    print("Loss: \n", loss.item())
print("Pred: \n", pred)


