import click
from tokenizers import Tokenizer


@click.command()
@click.option("--file", type=str)
@click.option("--text", type=str)
def main(file: str, text: str):
    tokenizer = Tokenizer.from_file(file)
    print(f"vocab_size = {tokenizer.get_vocab_size()}")

    output = tokenizer.encode(text)
    print(f"tokens = {output.tokens}")
    print(f"ids    = {output.ids}")


if __name__ == "__main__":
    main()
