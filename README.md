# tinyllm

## Preparation

Setup the environment using `rye`.

```shell
$ git clone https://github.com/teppchan/tinyllm.git
$ cd tinyllm
$ rye sync
```

## tinyllm

```shell
$ rye run python src/tinyllm/tokeni.py    # prepare the training dataset
$ rye run python src/tinyllm/train.py     # train the GPT model
$ rye run python src/tinyllm/inference.py # inference
```

## tinyllm2

Download `Aozora` dataset and train tokenizer with the dataset.

```shell
$ rye run python src/tinyllm2/prepare_aozora.py --book_num 10246 # download `aozora` dataset
$ rye run python src/tinyllm2/train_tokenizer.py --data_name aozora_10246  # train tokenizer with `aozora` dataset
```

Train the LLM model with `Aozora` dataset.
```shell
$ rye run python src/tinyllm2/train.py --data_name aozora_10246   
```

Generate texts.
```shell
$ rye run python src/tinyllm2/inference.py --tokenizer_file data/aozora_10246/tokenizer.json --model_file model/aozora_10246_20240723_201709/best_checkpoint.bin --text "こんにちは。"
```

## Reference 

- [ゼロから始める自作LLM｜Masayuki Abe (note.com)
](https://note.com/masayuki_abe/n/n365b500d91d2)
- [speed1313/jax-llm: JAX implementation of Large Language Models. You can train GPT-2-like model with 青空文庫 (aozora bunko-clean dataset) or any other text dataset. (github.com)
](https://github.com/speed1313/jax-llm)
- [青空文庫のテキストから作成したコーパスを Hugging Face で公開しました #LLM - Qiita](https://qiita.com/akeyhero/items/b53eae1c0bc4d54e321f)

