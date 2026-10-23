import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Optional, Dict
from pathlib import Path
from collections.abc import Mapping
import random

class TokenizedDataset(Dataset):
    def __init__(self, file_path: Union[str, Path], max_length: int):
        super().__init__()
        self.max_length = max_length
        self.tokens = np.memmap(file_path, dtype=np.int32, mode="r")
        self.size = len(self.tokens)//self.max_length - int(len(self.tokens)%self.max_length == 0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= self.size:
            raise IndexError("Index out of range")
        sequence = self.tokens[idx*self.max_length:(idx+1)*self.max_length+1]
        return {"input_ids": torch.from_numpy(sequence[:-1]), "labels": torch.from_numpy(sequence[1:])}

    def __len__(self) -> int:
        return self.size

def my_collate_fn(input_l):
    keys = input_l[0].keys()
    batch = {}
    for k in keys:
        batch[k] = torch.stack([input_l[idx][k] for idx in range(len(input_l))], dim=0)
    return batch

class DataLoader:
    def __init__(self,
                dataset: TokenizedDataset,
                batch_size: int,
                shuffle: bool,
                drop_last: bool = False,
                collate_fn = None,
                seed: Optional[int] = None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        if collate_fn is None:
            self.collate_fn = my_collate_fn
        else:
            self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.dataset)//self.batch_size + int(len(self.dataset)%self.batch_size != 0 and not self.drop_last)
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        batches = []
        if self.shuffle:
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(indices)
        
        for batch_idx in range(len(self)):
            start = batch_idx*self.batch_size
            end = min((batch_idx+1)*self.batch_size, len(self.dataset))
            yield self.collate_fn([self.dataset[indices[idx]] for idx in range(start, end)])
        
    