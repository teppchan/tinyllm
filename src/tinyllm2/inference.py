import torch
import numpy as np
from tokenizers import Tokenizer
import json
import click

from model import GPT

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
@click.option("--drop_out_rate", type=float, default=0.1)
@click.option("--layer_eps", type=float, default=1.0e-5)
@click.option("--text", type=str, default="これはなに？")
def main(
    tokenizer_file: str,
    model_file: str,
    embedding_size: int,
    num_heads: int,
    depth: int,
    sentence_size: int,
    drop_out_rate: float,
    layer_eps: float,
    text: str,
):
    checkpoint = torch.load(model_file, map_location=device)
    # print(checkpoint)
    tokenizer = Tokenizer.from_file(tokenizer_file)
    # print(tokenizer)

    tokenizer_json = json.load(open(tokenizer_file, "r"))
    vocab_size = len(tokenizer_json["model"]["vocab"])
    print(f"vocab_size = {vocab_size}")

    gpt = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_size,
        ffn_dim=embedding_size * 4,
        num_heads=num_heads,
        drop_out_rate=drop_out_rate,
        layer_eps=layer_eps,
        batch_first=True,
        T=sentence_size,
        N=depth,
    ).to(device)

    gpt.load_state_dict(checkpoint["model"])
    print(f"params = {gpt.calculate_params():,}")

    print(
        gpt.generate_sentence(
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
