import torch
import numpy as np
from tokenizers import Tokenizer
import json
import click

from model import Config, Transformer

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

@click.command()
@click.option("--tokenizer_file", type=str)
@click.option("--model_file", type=str)
@click.option("--embedding_size", type=int, default=768)
@click.option("--num_heads", type=int, default=6)
@click.option("--depth", type=int, default=6)
@click.option("--sentence_size", type=int, default=1024)
@click.option("--max_seq_len", type=int, default=1024)
@click.option("--text", type=str, default="これはなに？")
def main(
    tokenizer_file: str,
    model_file: str,
    embedding_size: int,
    num_heads: int,
    depth: int,
    sentence_size: int,
    max_seq_len: int,
    text: str,
):
    checkpoint = torch.load(model_file, map_location=device)
    # print(checkpoint)
    tokenizer = Tokenizer.from_file(tokenizer_file)
    # print(tokenizer)

    tokenizer_json = json.load(open(tokenizer_file, "r"))
    vocab_size = len(tokenizer_json["model"]["vocab"])
    print(f"vocab_size = {vocab_size}")

    conf = Config(
        dim=embedding_size,
        n_layers=depth,
        n_heads=num_heads,
        vocab_size=vocab_size,
        multiple_of=1,
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    print(conf)

    llama = Transformer(conf).to(device)

    llama.load_state_dict(checkpoint["model"])
    print(f"params = {llama.calculate_params():,}")

    print(
        llama.generate_sentence(
            text,
            sentence_size,
            128,
            tokenizer,
            device,
            top_K=20,
            temperature=2,
        )
    )

if __name__ == "__main__":
    main()
