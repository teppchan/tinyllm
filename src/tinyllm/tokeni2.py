
import pickle
import numpy as np
from tqdm import tqdm

with open("tokenized_dataset.bin", "rb") as p:
    tokenized = pickle.load(p)

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


