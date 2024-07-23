import torch
import tiktoken
import numpy as np

from model import GPT

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("best_checkpoint.bin", map_location="cuda")

tokenizer = tiktoken.get_encoding("gpt2")

sentence_size = 1024
embedding_size = 768
num_heads = 6
depth = 6

gpt = GPT(
    50257,
    embedding_size,
    embedding_size * 4,
    num_heads,
    0,
    batch_first=True,
    T=sentence_size,
    N=depth,
).to(device)


gpt.load_state_dict(checkpoint["model"])

print(
    gpt.generate_sentence(
        "What is OpenAI?",
        sentence_size,
        128,
        tokenizer,
        device,
        top_K=20,
        temperature=2,
    )
)
