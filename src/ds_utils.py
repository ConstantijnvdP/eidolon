"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
from typing import Callable, Dict, List, Tuple
from datasets import Dataset, Features, Value, arrow_dataset, concatenate_datasets 
import warnings
import jax.numpy as jnp
import jax.random as random

from log_fns import warning_format

"""
Build Dataset
"""
def merge_data_col(col_keys: List) -> Callable[[Dict], Dict]:
    return lambda data_col: {"input": [data_col[key] for key in col_keys]}


def preproc_motzkin_ds(ds, seq_len: int):
    data_keys = [str(i) for i in range(seq_len)]
    preproc_fn = merge_data_col(data_keys)

    return ds.map(preproc_fn, remove_columns=data_keys)

def samp_to_jnp(ds, start_idx, in_bs, out_bs, feature, dtype):
    arr = ds[start_idx: start_idx + in_bs][feature]
    return jnp.asarray(arr, dtype=dtype).reshape(out_bs)

'''
def get_stratified_val_idxs(neg_val_key, train_len, num_pos, ds_size, val_frac, shuf_pos_idxs):
    """
        Build a dataset outside of the train set that uses valid and invalid
        Motzkin chains proportionally to the set of all chains of the given length
    """
    #train_len, num_pos, ds_size = len_tup
    ## Take stratified sample of pos and neg data to build val set
    remainder_pos_idxs = shuf_pos_idxs[train_len:]
    shuffle_neg_idxs = random.permutation(neg_val_key, jnp.arange(num_pos, ds_size)).tolist()
    float_val_len = val_frac * (len(remainder_pos_idxs) + len(shuffle_neg_idxs))
    pos_frac = num_pos / ds_size
    neg_val_len = int((1-pos_frac) * float_val_len)
    neg_val_idxs = shuffle_neg_idxs[:neg_val_len]
    pos_val_len = int(float_val_len - neg_val_len)
    pos_val_idxs = remainder_pos_idxs[:pos_val_len]

    return pos_val_idxs + neg_val_idxs
'''


class MotzkinPipeline:
    def __init__(self, data_path: str, pos_ds_name: str, neg_ds_name: str, seq_len: int, train_size: int, train_pos_frac: float, test_size: int, test_pos_frac):
        """
        train_pos_frac is fraction of chains in the training dataset that are valid Motzkin chains
        For the validation step, all positive chains are used.
        test_pos_frac is the fraction of chains in the test set that are positive
        """
        assert train_pos_frac >= 0.0 and train_pos_frac <= 1.0, f"train_pos_frac should be a float in interval (0,1]; got {train_pos_frac}."
        assert test_pos_frac >= 0.0 and test_pos_frac <= 1.0, f"test_pos_frac should be a float in interval (0,1]; got {test_pos_frac}."
        self.seq_len = seq_len

        self.train_pos_len = int(train_pos_frac * train_size)
        self.train_neg_len = train_size - self.train_pos_len

        self.test_pos_len = int(test_pos_frac * test_size)
        self.test_neg_len = test_size - self.test_pos_len

        self.pos_ds_path = data_path + pos_ds_name
        self.neg_samp_ds_path = data_path + neg_ds_name
        self.ds_cache_dir = data_path + "/caches/"

    def get_train_test_idxs(self, key, pos_ds_size, neg_ds_size):
        assert self.train_pos_len + self.test_pos_len <= pos_ds_size, f"Number of valid chains in the train set, {self.train_pos_len}, plus test set, {self.test_pos_len}, exceeds number of valid data, {pos_ds_size}."
        assert self.train_neg_len + self.test_neg_len <= neg_ds_size, f"Number of invalid chains in the train set, {self.train_neg_len}, plus test set, {self.test_neg_len}, exceeds number of invalid data, {neg_ds_size}."
        ## Shuffle positive indices
        key, pos_key = random.split(key)
        shuf_pos_idxs = random.permutation(pos_key, pos_ds_size).tolist()       

        ## Shuffle negative indices
        key, neg_key = random.split(key)
        shuf_neg_idxs = random.permutation(neg_key, neg_ds_size).tolist()

        ## Get training data indices
        train_pos_idxs = shuf_pos_idxs[:self.train_pos_len]
        train_neg_idxs = shuf_neg_idxs[:self.train_neg_len]
        train_idxs = (train_pos_idxs, train_neg_idxs)

        if self.test_pos_len > pos_ds_size - self.train_pos_len:
            warnings.formatwarning = warning_format
            warnings.warn(
                f"""Number of desired positive chains for test set, {self.test_pos_len},
                is greater than the number remaining outside the train set, {len(remainder_pos_idxs)}.
                Only using all positive chains outside the train set."""
            )

        ## Get test data indices from indices outside train set
        test_pos_idxs = shuf_pos_idxs[self.train_pos_len:self.train_pos_len+self.test_pos_len]
        test_neg_idxs = shuf_neg_idxs[self.train_neg_len:self.train_neg_len+self.test_neg_len]
        test_idxs = (test_pos_idxs, test_neg_idxs)

        return train_idxs, test_idxs

    def get_ds(self, pos_ds, neg_ds, ds_idxs):
        pos_idxs, neg_idxs = ds_idxs
        ds = concatenate_datasets([pos_ds.select(pos_idxs), neg_ds.select(neg_idxs)])
        return preproc_motzkin_ds(ds, self.seq_len)

    def build(self, key):
        ## Get datasets
        pos_ds = Dataset.from_csv(self.pos_ds_path, cache_dir=self.ds_cache_dir)
        neg_samp_ds = Dataset.from_csv(self.neg_samp_ds_path, cache_dir=self.ds_cache_dir)

        train_idxs, test_idxs = self.get_train_test_idxs(key, len(pos_ds), len(neg_samp_ds))

        train_ds = self.get_ds(pos_ds, neg_samp_ds, train_idxs)
        val_ds = preproc_motzkin_ds(pos_ds, self.seq_len)
        test_ds = self.get_ds(pos_ds, neg_samp_ds, test_idxs)
    
        return train_ds, val_ds, test_ds


## Build Iterator
def get_jnp_batch_ds_iter(dataset: arrow_dataset.Dataset, batch_shape: Tuple[int], dtype: jnp.dtype = jnp.int32, shuffle_seed=None):
    assert len(batch_shape) == 3, "Batch shape must have three elements (num dev, per dev batch size, seq_len)!"
    batch_size = batch_shape[0] * batch_shape[1]
    if batch_size > len(dataset):
        print(f"""Effective batch size {batch_size} (num dev * per_dev_bs)
                  larger than dataset {len(dataset)}! Scaling down to match.""")
        batch_size = len(dataset)

    ## Number of full batches (num_dev x per_dev_batch_size)
    full_batches = len(dataset) // batch_size
    ## Number of data points that fill the full batches
    num_full_data = full_batches * batch_size

    ## Number of leftover data lines
    leftover_data = len(dataset) - num_full_data
    ## per device batch size that captures most remaining data points evenly
    leftover_bs = leftover_data // batch_shape[0]

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
        #dataset = shuffle_ds(dataset, shuffle_seed)

    def jnp_batch_ds_iter():
        for i in range(0, num_full_data, batch_size):
            in_arr = samp_to_jnp(dataset, i, batch_size, batch_shape, "input", dtype)
            lab_arr = samp_to_jnp(dataset, i, batch_size, batch_shape[:2], "label", dtype)

            yield {"input": in_arr, "label": lab_arr}

        if leftover_bs > 0:
            in_arr = samp_to_jnp(dataset, num_full_data, leftover_bs, (batch_shape[0], -1, batch_shape[2]), "input", dtype)
            lab_arr = samp_to_jnp(dataset, num_full_data, leftover_bs, (batch_shape[0], -1), "label", dtype)
            yield {"input": in_arr, "label": lab_arr}

    return jnp_batch_ds_iter
