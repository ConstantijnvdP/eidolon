"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import os
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state, checkpoints
from flax import jax_utils

from mps import DenseMPS, get_sample_generator
from optimizer import warmup_constant_schedule, get_sgd, get_adam

checkpoint_dir = os.getcwd() + "/outputs/test_checkpoint"

num_dev = 1
bs = 1024
mps_cfg = {
    'num_cores': 16,
    'vocab_size':3,
    'bond_dim': 8,
    'embed_dtype':jnp.float32,
    'partial_len': 1
}
sched_cfg = {
    'init_value':0.00001,
    'end_value':0.001,
    'warmup_steps':500
}
opt_cfg = {
    'momentum':0.0
}
RNG_key = random.PRNGKey(0)

model = DenseMPS(**mps_cfg)
sample_generator = get_sample_generator(model, mps_cfg['bond_dim'])
init_array = jnp.ones((num_dev, bs, mps_cfg['num_cores']), dtype=jnp.int32)
params = model.init(RNG_key, init_array)['params']

scheduler = warmup_constant_schedule(**sched_cfg)
optimizer = get_sgd(opt_cfg, scheduler)

simulacrum = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

load_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=simulacrum)
print(load_state.params)

#for samp in sample_generator(load_state.params, RNG_key, 10):
#    print(samp)
