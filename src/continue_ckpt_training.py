"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import os
import omegaconf
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils

from mps import DenseMPS, get_sample_generator
from optimizer import warmup_constant_schedule, get_sgd, get_adam
from main import build_model

@hydra.main(config_path='conf', config_name='chkpt_cfg')
def main(cfg: DictConfig, run_cfg: DictConfig, run_dir: str, chkpt_num: int) -> None:
    checkpoint = run_dir + f"checkpoints/checkpoint_{chkpt_num}"
    
    num_dev = jax.local_device_count()
    seq_len = cfg.hparams.seq_len
    init_shape = (num_dev, 1, seq_len)
    RNG_key = random.PRNGKey(0)

    init_state, _, _, scheduler = build_model(RNG_key, cfg, init_shape)
    state = jax_utils.replicate(init_state)
    load_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint, target=state)


if __name__ == "__main__":
    run_dir = input("Input directory of run to continue: ")
    chkpt_num = int(input("Input checkpoint number: "))
    cfg_path = run_dir + "/hydra/config.yaml"
    run_cfg = omegaconf.OmegaConf.load(cfg_path)
    main(run_cfg, run_dir, chkpt_num)
