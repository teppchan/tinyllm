# from tokenizers import Tokenizer
import datetime
import gc
import json
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Config, Transformer

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("--data_name", type=str)
@click.option("--logdir", type=str, default="logs")
@click.option("--max_iters", type=int, default=10000)
@click.option("--batch_size", type=int, default=4)
@click.option("--batch_iteration", type=int, default=128)
@click.option("--embedding_size", type=int, default=768)
@click.option("--num_heads", type=int, default=6)
@click.option("--depth", type=int, default=6)
@click.option("--sentence_size", type=int, default=1024)
@click.option("--max_seq_len", type=int, default=1024)
def main(
    data_name: str,
    logdir: str,
    max_iters: int,
    batch_size: int,
    batch_iteration: int,
    embedding_size: int,
    num_heads: int,
    depth: int,
    sentence_size: int,
    max_seq_len: int,
):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_filename = Path(f"data/{data_name}/tokenized_text.bin")
    tokenizer_path = Path(f"data/{data_name}/tokenizer.json")
    model_path = Path(f"model/{data_name}_{now}")

    model_path.mkdir(exist_ok=True, parents=True)

    print(f"load tokenizer ({tokenizer_path})")
    tokenizer_json = json.load(open(tokenizer_path, "r"))
    vocab_size = len(tokenizer_json["model"]["vocab"])
    print(f"vocab_size = {vocab_size}")

    with open(data_filename, "rb") as f:
        tokenized_text = f.read()
    tokenized_text = np.frombuffer(tokenized_text, dtype=np.int64)
    total_tokens = len(tokenized_text)
    print(f"total_token = {total_tokens}")

    train_ratio = 0.95
    split_idx = int(train_ratio * total_tokens)
    train_data = tokenized_text[:split_idx]
    val_data = tokenized_text[split_idx:]

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
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)

    conf = Config(
        dim=embedding_size,
        n_layers=depth,
        n_heads=num_heads,
        vocab_size=vocab_size,
        multiple_of=1,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    print(conf)

    llama = Transformer(conf).to(device)

    print(llama)

    writer = SummaryWriter(log_dir=f"{logdir}/{data_name}_{now}")
    # x, y = get_batch("train", batch_size=batch_size, device=device)
    # writer.add_graph(llama, (x, 0))

    optimizer = torch.optim.AdamW(llama.parameters(), lr=1e-4)

    best_loss = 1e9
    begin = 0
    val_iteration = 1

    gc.collect()
    torch.cuda.empty_cache()

    llama.train()
    for cur_iter in tqdm(range(begin, max_iters)):
        train_loss = 0.0
        for b in range(batch_iteration):
            optimizer.zero_grad()

            x, y = get_batch("train", batch_size=batch_size, device=device)
            pred = llama(x)
            loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)), y.view(-1), ignore_index=-1
            )
            train_loss += loss.detach().cpu().item()

            loss.backward()
            optimizer.step()

            del x, y, loss, pred
            gc.collect()
            torch.cuda.empty_cache()

        writer.add_scalar("loss/train", train_loss / batch_iteration, cur_iter)

        valid_loss = 0.0
        for _ in range(val_iteration):
            with torch.no_grad():
                x, y = get_batch("valid", batch_size=batch_size, device=device)
                pred = llama(x)
                loss = F.cross_entropy(
                    pred.view(-1, pred.size(-1)), y.view(-1), ignore_index=-1
                )
                valid_loss += loss.detach().cpu()
                del x, y, loss, pred

        avg_valid_loss = valid_loss.item() / val_iteration
        writer.add_scalar("loss/valid", avg_valid_loss, cur_iter)

        if best_loss > avg_valid_loss:
            best_loss = avg_valid_loss
            checkpoint = {
                "model": llama.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": cur_iter,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, f"{model_path}/best_checkpoint.bin")

        if torch.isnan(valid_loss):
            print("Loss is Nan! ;(")
            break

        checkpoint = {
            "model": llama.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": cur_iter,
            "best_loss": best_loss,
            "loss": avg_valid_loss,
        }
        torch.save(checkpoint, f"{model_path}/latest_checkpoint.bin")

        with open(f"{model_path}/learning_detail_latest.txt", "w") as f:
            f.write("Training condition:\n")
            f.write(f"iter: {cur_iter}\n")
            f.write("hyper params:\n")
            f.write(f"vocab_size: {vocab_size}, ")
            f.write(f"embedding size: {embedding_size}, ")
            f.write(f"ffn: {embedding_size*4}, ")
            f.write(f"num_heads: {num_heads}, ")
            f.write(f"Depth: {depth}, ")
            f.write(f"sentence_size: {sentence_size}\n")
            f.write(f"val_loss: {avg_valid_loss}\n")

        del valid_loss

        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
