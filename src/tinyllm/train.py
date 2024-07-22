import gc
from time import sleep

import numpy as np
import tiktoken
import torch
from model import GPT
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")

sentence_size = 1024
batch_size = 6

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(
    split: str,
    batch_size=batch_size,
    device=device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data

    index = torch.randint(len(data) - sentence_size, (batch_size,))

    x = torch.stack(
        [
            torch.from_numpy((data[i : i + sentence_size]).astype(np.int64))
            for i in index
        ]
    )

    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + sentence_size]).astype(np.int64))
            for i in index
        ]
    )

    if device == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True)
        )
    return x.to(device), y.to(device)


embedding_size = 768
num_heads = 6
tokenizer = tiktoken.get_encoding("gpt2")

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

warmup_iters = 2000

optimizer = torch.optim.Adam(gpt.parameters(), lr=0.0001)

max_lr = 2.5e-5
min_lr = 2.5e-6
max_iters = 10000


def get_lr(cur_iter):
    if cur_iter < warmup_iters:
        return max_lr * cur_iter / warmup_iters
    return (max_lr * (np.cos(cur_iter / max_iters * np.pi) + 1)).clip(min_lr, max_lr)


batch_iteration = 128
scaler = torch.cuda.amp.GradScaler(enabled=True)
best_loss = 1e9
begin = 0
val_iteration = 1

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda cur_iter: get_lr(cur_iter))
gc.collect()
torch.cuda.empty_cache()

gpt.train()
for cur_iter in tqdm(range(begin, max_iters)):
    scheduler.step()

    for batch_iter in range(batch_iteration):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            x, y = get_batch("train", batch_size=batch_size, device=device)
            padding_mask, mask = gpt.create_mask(x, 0, device)
            loss, pred = gpt(x, y, padding_mask, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        del x, y, padding_mask, mask, loss, pred

    valid_loss = 0
    for val_iter in range(val_iteration):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                x, y = get_batch("valid", batch_size=batch_size, device=device)
                padding_mask, mask = gpt.create_mask(x, 0, device)
                loss, pred = gpt(x, y, padding_mask, mask)
                valid_loss += loss.detach()

                del x, y, padding_mask, mask, loss, pred

    avg_valid_loss = valid_loss.item() / val_iteration
    if best_loss > avg_valid_loss:
        best_loss = avg_valid_loss
        checkpoint = {
            "model": gpt.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "iter": cur_iter,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, "best_checkpoint.bin")
        print(f"params updated. Best Loss: {best_loss}")
        print(f"Val all loss: {avg_valid_loss}")

    if torch.isnan(valid_loss):
        print("Loss is Nan! ;(")
        break

    checkpoint = {
        "model": gpt.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iter": cur_iter,
        "best_loss": best_loss,
        "loss": avg_valid_loss,
    }
    torch.save(checkpoint, "latest_checkpoint.bin")

    with open("learning_detail_latest.txt", "w") as f:
        f.write("Training condition:\n")
        f.write(f"iter: {cur_iter}\n")
        f.write("hyper params:\n")
        f.write("vocab_size: 50257, ")
        f.write(f"embedding size: {embedding_size}, ")
        f.write(f"ffn: {embedding_size*4}, ")
        f.write(f"num_heads: {num_heads}, ")
        f.write(f"Depth: {depth}, ")
        f.write(f"sentence_size: {sentence_size}\n")
        f.write(f"lr: {scheduler.get_last_lr()[0]}, best_loss: {best_loss}\n")
        f.write(f"val_loss: {avg_valid_loss}\n")

    del valid_loss






