from pathlib import Path

import click
from datasets import load_dataset


@click.command()
@click.option("--book_num", type=int, default=10)
def main(book_num: int):
    data_name = f"aozora_{book_num}"

    dir = Path(f"data/{data_name}")
    dir.mkdir(exist_ok=True, parents=True)
    save_path = dir / "input.txt"
    ds = load_dataset(
        path="globis-university/aozorabunko-clean",
        cache_dir="data/aozora/cache",
    )
    ds = ds.filter(lambda row: row["meta"]["文字遣い種別"] == "新字新仮名")
    print(f"{book_num} books out of {len(ds['train'])} are used")
    with open(save_path, "w") as f:
        for i, book in enumerate(ds["train"]):
            if i > book_num:
                break
            f.write(book["text"])
            f.write("<|endoftext|>")


if __name__ == "__main__":
    main()
