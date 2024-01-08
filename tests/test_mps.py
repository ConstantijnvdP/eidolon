"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import functools
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap
from flax import jax_utils
import pytest
from numpy_mps_john import pair_with_one_hot, mps_norm

import os
import sys
## Add src/ to python path to import functions
file_path = os.path.dirname(__file__)
file_path_list = file_path.split(os.sep)
src_path_list = file_path_list[:-1] + ['src']
src_path = f"{os.sep}".join(src_path_list)
sys.path.append(src_path)
from motzkin_ds import gen_all_spin_chains
from train_eval_utils import get_mps_grad_fn
from main import build_model, factorize_dense_model


precision = jax.lax.Precision.HIGHEST
matmul = functools.partial(jnp.matmul, precision=precision)
einsum = functools.partial(jnp.einsum, precision=precision)

num_dev = jax.local_device_count()
vocab_size = 3
seq_len = 8
dense_bond_dim = 8
partial_len = seq_len 
scaled_error_threshold = 1E-6   ## contraction calculations must be at least this close

@pytest.fixture
def common_model_cfg():
    global seq_len, num_dev
    batch_size = 1
    init_batch_shape = (batch_size, seq_len)

    optim_cfg = {
        'name': 'sgd',
        'params': {'momentum': 0.9, 'nesterov': False}
    }
    sched_cfg = {
        'name': 'warmup_constant',
        'params': {'init_value': 0.00001, 'end_value': 0.0001, 'warmup_steps': 500}
    }

    return (optim_cfg, sched_cfg, init_batch_shape)

@pytest.fixture
def test_batch():
    global vocab_size, seq_len, num_dev
    batch_size = 4
    batch_shape = (num_dev, batch_size, seq_len)
    test_key = jax.random.PRNGKey(0)
    batch = jax.random.randint(test_key, batch_shape, 0, vocab_size)
    return batch

@pytest.fixture
def single_test_input(test_batch):
    global seq_len
    return test_batch[0, 0]

def all_chains_jnp_iter(n, batch_size):
    """
    n = sequence length
    """
    num_chains = 3**n
    data_gen = gen_all_spin_chains(n)
    if num_chains % batch_size == 0:
        num_batches = num_chains / batch_size
    else:
        num_batches = (num_chains//batch_size)+1 ## includes final incomplete batch

    for _ in range(num_batches):
        iter_slice = itertools.islice(data_gen, batch_size)
        yield jnp.asarray(list(iter_slice), dtype=jnp.int32)


dense_cfg = {
    'name': 'dense_mps',
    'init_params': {
        'num_cores': seq_len,
        'vocab_size': vocab_size,
        'h_bond_dim': dense_bond_dim,
        'embed_dtype': jnp.float32,
        'partial_len': partial_len,
        'boundary_var': 0.5,
        'internal_var': 0.1
    }
}

factored_cfg = {
    'name': 'factored_mps',
    'init_params': dense_cfg['init_params'].copy()
}
## Make sure h_bond_dim ** core_height = dense_bond_dim!!
factored_cfg['init_params']['core_height'] = 3
factored_cfg['init_params']['h_bond_dim'] = 2
factored_cfg['init_params']['v_bond_dim'] = 4
factored_cfg['init_params']['skip_connection'] = False


## Dense model fixtures
@pytest.fixture
def dense_model(common_model_cfg):
    init_key = jax.random.PRNGKey(0)

    state, LNS_fn, _, _ = build_model(init_key, dense_cfg, *common_model_cfg)

    return (state, LNS_fn)

@pytest.fixture
def dense_cores(dense_model):
    global seq_len
    state, _ = dense_model
    return [state.params[f"core_{i}"]['embedding'] for i in range(seq_len)]

@pytest.fixture
def dense_output(dense_model, single_test_input):
    state, _ = dense_model
    print("Dense")
    return state.apply_fn({'params': state.params}, single_test_input.reshape(1,-1))

@pytest.fixture
def dense_norm(dense_model):
    state, LNS_fn = dense_model
    LNS = LNS_fn(state.params)
    return jnp.sqrt(jnp.exp(LNS))


## Factored model fixtures
@pytest.fixture
def factored_model(common_model_cfg):
    global factored_cfg
    init_key = jax.random.PRNGKey(0)
    state, LNS_fn, _, _ = build_model(init_key, factored_cfg, *common_model_cfg)
    return (state, LNS_fn)

@pytest.fixture
def factorized_output(dense_model, factored_model, single_test_input):
    """
    Factorize "dense_model" and calculate the test output
    """
    global factored_cfg
    donor_state, _ = dense_model
    donor_params = donor_state.params
    params = factorize_dense_model(donor_params, factored_cfg)
    state, _ = factored_model
    print("Factored")
    return state.apply_fn({'params': state.params}, single_test_input.reshape(1,-1))

@pytest.fixture
def factored_state_norm(factored_model):
    """
    Build a factored core model and calculate its norm
    """
    state, LNS_fn = factored_model
    LNS = LNS_fn(state.params)
    norm = jnp.sqrt(jnp.exp(LNS))
    return (state, norm)

@pytest.fixture
def skip_factored_state_norm(common_model_cfg):
    global factored_cfg
    factored_cfg['init_params']['skip_connection'] = True

    init_key = jax.random.PRNGKey(0)
    state, LNS_fn, _, _ = build_model(init_key, factored_cfg, *common_model_cfg)
    LNS = LNS_fn(state.params)
    norm = jnp.sqrt(jnp.exp(LNS))
    return (state, norm)


## John's numpy model fixtures
@pytest.fixture
def john_model(dense_cores):
    global vocab_size, seq_len, dense_bond_dim
    bond_dim = dense_bond_dim
    john_mps = [np.array(dense_cores[i]) for i in range(seq_len)]
    for i in range(1, seq_len-1):
        john_mps[i] = john_mps[i].reshape(vocab_size, bond_dim, bond_dim).transpose(1, 0, 2)
    john_mps[-1] = john_mps[-1].T

    return john_mps


## Tests
def calc_scaled_error(x, y):
    """
    take the absolutel value of the error between two items,
    and scale it by their average
    """
    return np.exp( np.log(2*np.abs(x-y) + 1E-8) - np.log(x+y) )

## Contractions
einsum_output = jnp.zeros(1)
def test_contract_einsum(dense_output, dense_cores, single_test_input):
    """
    compare dense core model output to einsum calculation
    """
    global seq_len, dense_bond_dim, einsum_output
    bond_dim = dense_bond_dim

    einsum_output = dense_cores[0][single_test_input[0]]
    for i in range(1, seq_len-1):
        ith_core = dense_cores[i][single_test_input[i]].reshape(bond_dim, bond_dim)
        einsum_output = einsum("i,ij->j", einsum_output, ith_core)
    Rcore = dense_cores[-1][single_test_input[-1]]
    einsum_output = einsum("i,i", einsum_output, Rcore)
    log_einsum_output = np.log(np.abs(einsum_output))

    scaled_error = calc_scaled_error(dense_output, log_einsum_output)
    print("Einsum contraction scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold

john_output = jnp.zeros(1)
def test_john_contract(single_test_input, john_model, dense_output):
    """
    compare dense core model output to John's numpy's code output
    """
    global john_output
    john_output = pair_with_one_hot(john_model, single_test_input)
    log_john_output = np.log(np.abs(john_output))

    scaled_error = calc_scaled_error(dense_output, log_john_output)
    print("John contraction scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold

def test_tests(dense_cores, test_batch, john_model):
    global einsum_output, john_output
    scaled_error = calc_scaled_error(einsum_output, john_output)
    print("Einsum vs. John contraction scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold

def test_factorize_dense_model(dense_output, factorized_output):
    scaled_error = calc_scaled_error(dense_output, factorized_output)
    print("Dense vs. Factorized Dense scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold


## Norms
def get_core_list(params, seq_len):
    return [params[f'core_{i}']['embedding'] for i in range(seq_len)]

def test_einsum_norm(dense_norm, dense_model):
    """
    compare dense core norm calculation to einsum calculation
    """
    global vocab_size, seq_len, dense_bond_dim
    state, _ = dense_model
    bond_dim = dense_bond_dim
    einsum_cores = get_core_list(state.params, seq_len)

    cap = einsum("vt,vb->bt", einsum_cores[0], einsum_cores[0])
    for i in range(1, seq_len-1):
        core = einsum_cores[i].reshape(vocab_size, bond_dim, bond_dim)
        ct = einsum("bt,vtr->vbr", cap, core)
        cap = einsum("vmb,vmt->bt", core, ct)
    ct = einsum("bt,vt->vb", cap, einsum_cores[-1])
    norm_sq = einsum("vb,vb", ct, einsum_cores[-1])

    scaled_error = calc_scaled_error(dense_norm, jnp.sqrt(norm_sq))
    print("Einsum norm scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold

def test_john_norm(dense_norm, john_model):
    """
    compare dense core norm calculation to John's numpy's code calculation
    """
    norm = mps_norm(john_model)
    scaled_error = calc_scaled_error(dense_norm, norm)
    print("John norm scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold


def sum_all_contractions_error(state, norm, n=8, batch_size=1024):
    print(f"\nCalculating total unnormalized prob mass for all chains of length {n}.")
    tot_output = 0.0
    for batch in all_chains_jnp_iter(n, batch_size):
        log_outputs = state.apply_fn({'params': state.params}, batch)
        tot_output += jnp.exp(2*log_outputs).sum()

    manual_norm = jnp.sqrt(tot_output)
    scaled_error = calc_scaled_error(norm, manual_norm)
    return scaled_error

def test_dense_sum_contractions_norm(dense_model, dense_norm):
    """
    test that summing the results of contracting all inputs with the dense model equals the norm calculation
    """
    state, _ = dense_model
    scaled_error = sum_all_contractions_error(state, dense_norm)
    print("Dense manual norm scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold

def test_factored_sum_contractions_norm(factored_state_norm):
    """
    test that summing the results of contracting all inputs with the factored core model equals its norm calculation
    """
    scaled_error = sum_all_contractions_error(*factored_state_norm)
    print("Factored manual norm scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold

def test_skip_factored_sum_contractions_norm(skip_factored_state_norm):
    """
    test that summing the results of contracting all inputs with the factored core model with skip connection equals its norm calculation
    """
    scaled_error = sum_all_contractions_error(*skip_factored_state_norm)
    print("Skip factored manual norm scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold


## Grads
def test_grad_batching(dense_model, test_batch):
    """
    test that code with Jax's device parallelism function (pmap) isn't distorting gradients
    """
    num_dev, batch_size, seq_len = test_batch.shape
    state, LNS_fn = dense_model
    p_state = jax_utils.replicate(state)

    test_labels = jnp.ones(test_batch.shape[:2], dtype=jnp.int32)
    batch = {
        'input': test_batch,
        'label': test_labels
    }

    ## N.B. only need up until sync_grads in defining train step
    alpha = 0.0
    grad_fn = get_mps_grad_fn(alpha, LNS_fn)
    axis_name = 'num_devices'
    def step(state, batch):
        """Train for a single step."""
        (p_loss, p_log_norm_sq), p_grads = grad_fn(state.params, state, batch)
        return jax.lax.pmean(p_grads, axis_name=axis_name)

    p_train_step = pmap(step, axis_name=axis_name)

    sync_grads = p_train_step(p_state, batch)
    ## return dict of 'core' 'embedding'

    man_grads = jax.tree_map(lambda x: x*0, sync_grads)
    for dev_idx in range(num_dev):
        for batch_idx in range(batch_size):
            datum = {
                'input': test_batch[dev_idx, batch_idx].reshape(1, -1),
                'label': test_labels[dev_idx, batch_idx].reshape(1, -1)
            }
            _, grad = grad_fn(state.params, state, datum)
            man_grads = jax.tree_map(lambda x, y: x+y, man_grads, grad)

    man_grads = jax.tree_map(lambda x: x/(num_dev*batch_size), man_grads)

    sum_grad = 0
    grad_error = 0
    for i in range(seq_len):
        sum_grad += (sync_grads[f'core_{i}']['embedding'] + man_grads[f'core_{i}']['embedding']).sum()
        grad_error += jnp.abs(sync_grads[f'core_{i}']['embedding'] - man_grads[f'core_{i}']['embedding']).sum()

    scaled_error = 2*grad_error/sum_grad
    print("Batch gradients scaled error:", scaled_error)
    assert scaled_error < scaled_error_threshold
