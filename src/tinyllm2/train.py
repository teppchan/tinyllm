import gc
import json
import click
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
from tokenizers import Tokenizer
import datetime

from torch.utils.tensorboard import SummaryWriter

from model import GPT

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option(
    "--data_name",
    type=str,
)
@click.option("--logdir", type=str, default="logs")
@click.option("--max_iters", type=int, default=10000)
@click.option("--batch_size", type=int, default=4)
@click.option("--batch_iteration", type=int, default=128)
@click.option("--embedding_size", type=int, default=768)
@click.option("--num_heads", type=int, default=6)
@click.option("--depth", type=int, default=6)
@click.option("--sentence_size", type=int, default=1024)
@click.option("--drop_out_rate", type=float, default=0.1)
@click.option("--layer_eps", type=float, default=1.0e-5)
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
    drop_out_rate: float,
    layer_eps: float,
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

    # tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # print(train_data[:100])
    # print(tokenizer.decode(train_data[:100]))
    # exit()

    # print(train_data)
    # exit()

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

    gpt = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_size,
        ffn_dim=embedding_size * 4,
        num_heads=num_heads,
        drop_out_rate=drop_out_rate,
        batch_first=True,
        T=sentence_size,
        N=depth,
        layer_eps=layer_eps,
    ).to(device)

    print(gpt)

    writer = SummaryWriter(log_dir=f"{logdir}/{data_name}_{now}")
    x, y = get_batch("train", batch_size=batch_size, device=device)
    padding_mask, mask = gpt.create_mask(x, 0, device)
    writer.add_graph(gpt, (x, y, padding_mask, mask))

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-4)

    # batch_iteration = 128
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_loss = 1e9
    begin = 0
    val_iteration = 1

    gc.collect()
    torch.cuda.empty_cache()

    gpt.train()
    for cur_iter in tqdm(range(begin, max_iters)):
        train_loss = 0.0
        for _ in range(batch_iteration):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                x, y = get_batch("train", batch_size=batch_size, device=device)
                padding_mask, mask = gpt.create_mask(x, 0, device)
                loss, pred = gpt(x, y, padding_mask, mask)
                train_loss += loss.detach()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # print(f"x = {x}")
            # print(f"y = {y}")
            # print(f"padding_mask = {padding_mask}")
            # print(f"mask = {mask}")
            # print(f"loss = {loss}")
            # print(f"pred = {pred}")

            del x, y, padding_mask, mask, loss, pred

        writer.add_scalar("loss/train", train_loss.item() / batch_iteration, cur_iter)

        valid_loss = 0.0
        for _ in range(val_iteration):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    x, y = get_batch("valid", batch_size=batch_size, device=device)
                    padding_mask, mask = gpt.create_mask(x, 0, device)
                    loss, pred = gpt(x, y, padding_mask, mask)
                    valid_loss += loss.detach()
                    # print(f"x = {x}")
                    # print(f"y = {y}")
                    # print(f"padding_mask = {padding_mask}")
                    # print(f"mask = {mask}")
                    # print(f"loss = {loss}")
                    # print(f"pred = {pred}")

                    del x, y, padding_mask, mask, loss, pred

        avg_valid_loss = valid_loss.item() / val_iteration
        writer.add_scalar("loss/valid", avg_valid_loss, cur_iter)

        if best_loss > avg_valid_loss:
            best_loss = avg_valid_loss
            checkpoint = {
                "model": gpt.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "iter": cur_iter,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, f"{model_path}/best_checkpoint.bin")
            # print(f"params updated. Best Loss: {best_loss}")
            # print(f"Val all loss: {avg_valid_loss}")

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
        torch.save(checkpoint, f"{model_path}/latest_checkpoint.bin")

        with open(f"{model_path}/learning_detail_latest.txt", "w") as f:
            f.write("Training condition:\n")
            f.write(f"iter: {cur_iter}\n")
            f.write("hyper params:\n")
            f.write("vocab_size: 50257, ")
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
