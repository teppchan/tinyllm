import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
tokenizer.encode_ordinary("This is a sample.")

from datasets import load_dataset

num_proc_load_dataset = 8
dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
print(dataset)


import pickle
with open("dataset.bin", "wb") as p:
    pickle.dump(dataset, p)

split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset["val"] = split_dataset.pop("test")
print(split_dataset)

def process(example):
    ids = tokenizer.encode_ordinary(example["text"])
    ids.append(tokenizer.eot_token) # 末尾に<endoftext>Tokenを追加

    out = {"ids": ids, "len": len(ids)}
    return out

tokenized = split_dataset.map(
    process,
    remove_columns=["text"],
    desc="tokenizing the split",
    num_proc=num_proc_load_dataset,
)
print(tokenized)

with open("tokenized_dataset.bin", "wb") as p:
    pickle.dump(tokenized, p)

import numpy as np
from tqdm import tqdm
for split, dset in tokenized.items():
    filename = split+".bin"
    length = np.sum(dset["len"], dtype=np.uint64)
    write_data = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(length, ))
    iteration = 1024
    index = 0
    for iter_index in tqdm(range(iteration)):
        add_data = dset.shard(num_shards=iteration, index=iter_index, contiguous=True).with_format("numpy")
        add_data = np.concatenate(add_data["ids"])
        add_length=len(add_data)
        write_data[index:index+add_length] = add_data
        index += add_length
    write_data.flush()


