"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import os
from tqdm import tqdm
from typing import List
import csv
from jax import random
import jax.numpy as jnp

codex = [")", "-", "("]

val0 = jnp.array([2,1,1,1,0])
val1 = jnp.array([1,1,1,2,0])
val2 = jnp.array([1,2,1,0,1])

bad0 = jnp.array([2,1,1,0,2])
bad1 = jnp.array([2,1,1,0,0])
bad2 = jnp.array([0,1,1,2,0])
bad3 = jnp.array([2,2,1,1,0])

val_list = [val0, val1, val2]
bad_list = [bad0, bad1, bad2, bad3]


def decode_samp(samp):
    return "".join([codex[i] for i in samp])


def valid_motzkin(samp):
    chk_sum = 0
    for i in samp:
        chk_sum += i-1
        if chk_sum < 0:
            return False
        else:
            continue
    if chk_sum == 0:
        return True
    else:
        return False


## Synthesize Data
## Generating data points (Lists) to save to CSV
def increment_chain(chain: List) -> List:
    assert all(spin >= 0 and spin < 3 for spin in chain), f"Invalid spins in spin chain: {chain} (must have 0 <= spin < 3)"
    new = chain.copy()
    inc_idx = -1
    carryover = True
    while carryover:
        new[inc_idx] = (new[inc_idx] + 1) % 3
        if new[inc_idx] == 0:  ## end was 2
            inc_idx -= 1  ## Move left a digit and continue incrementing
            if inc_idx == -len(chain)-1:
                ## Don't increment beyond length of input chain
                break
        else:
            carryover = False
    return new


def gen_all_spin_chains(seq_len):
    chain = [2]*seq_len
    for _ in tqdm(range(3 ** seq_len), desc="Chain"):
        chain = increment_chain(chain)
        yield chain


def gen_motzkin_csv(seq_len, filename, valid_only):
    with open(filename, 'w') as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        header = ["label"] + list(range(seq_len))
        csvwriter.writerow(header)

        for chain in gen_all_spin_chains(seq_len):
            is_valid = valid_motzkin(chain)
            if valid_only:
                if is_valid:
                    datum = [int(is_valid)] + chain
                    csvwriter.writerow(datum)
            else:
                datum = [int(is_valid)] + chain
                csvwriter.writerow(datum)

def gen_invalid_motzkin_samples_csv(seq_len, filename, save_prob, seed=0):
    key = random.PRNGKey(seed)
    with open(filename, 'w') as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        header = ["label"] + list(range(seq_len))
        csvwriter.writerow(header)

        for chain in gen_all_spin_chains(seq_len):
            is_valid = valid_motzkin(chain)
            if not is_valid:
                key, sample_key = random.split(key)
                if random.uniform(sample_key) <= save_prob:
                    datum = [int(is_valid)] + chain
                    csvwriter.writerow(datum)


if __name__ == "__main__":
    seq_len = int(input(
        "Please enter sequence length of Motzkin dataset to generate: "))

    need_input = True
    while need_input:
        valid_only_input = input("Save only valid Motzkin chains? (Y/n): ")
        if valid_only_input == "Y":
            valid_only = True
            need_input = False
        elif valid_only_input == "n":
            valid_only = False
            need_input = False
        else:
            print("Please enter Y or n")

    curr_dir = os.getcwd()
    curr_dir_list = os.path.normpath(curr_dir).split(os.sep)
    root_dir = f"{os.sep}".join(curr_dir_list[:-1]) ## Root of code dir, parent of src, outputs, data, &c.
    data_dir = root_dir + "/data/"

    ## Generate only valid samples
    if valid_only:
        filename = data_dir + f"{seq_len}_motzkin_data_valid.csv"
    else:
        filename = data_dir + f"{seq_len}_motzkin_data.csv"

    print(f"File will be saved to {filename}")
    gen_motzkin_csv(seq_len, filename, valid_only)
