"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import functools
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

#precision = jax.lax.Precision.HIGHEST
precision = jax.lax.Precision.DEFAULT
matmul = functools.partial(jnp.matmul, precision=precision)
einsum = functools.partial(jnp.einsum, precision=precision)

## Parameter initialization functions
def noisy_zeros_init(init_variance):
    def init(key, shape, dtype=jnp.float_):
        return random.normal(key, shape, dtype) * init_variance
    return init

def noisy_unit_ones_init(init_variance):
    def init(key, shape, dtype=jnp.float_):
        base = jnp.ones(shape, dtype) / np.sqrt(shape[-1])  # divide by number of elements to "normalize"
        noise = random.normal(key, shape, dtype) * init_variance
        return base + noise
    return init

def noisy_identity_init(init_variance):
    def init(key, shape, dtype=jnp.float_):
        bond_dim = np.sqrt(shape[1]).astype(int)
        base = jnp.identity(bond_dim, dtype).flatten()
        noise = random.normal(key, shape, dtype) * init_variance
        return base + noise
    return init

'''
def noisy_orth_init(init_variance):
    def init(key, shape, dtype=jnp.float_):
        rand = jnp.random.uniform(key, shape, dtype)
        base = jnp.linalg.qr(rand)[0]
        noise = random.normal(key, shape, dtype) * init_variance
'''

def factorize_core(name, params, h_bond_dim, v_bond_dim, height, num_h_legs, skip_connection=False, aug_key=jax.random.PRNGKey(0), extend_minval=0.001, extend_maxval=0.01):
    """
    Input: DenseCore's parameters as input
        Shape: (vocab_size, dense_bond_dim^2)
        dense bond dimension = h_bond_dim ** height
    Output: (Skip)FactorizedCore's parameters
        Output is pytree
    """
    assert len(params.shape) == 2, "Input params must have two indices: (vocab_size, dense_bond_dim**2)."
    assert num_h_legs == 1 or num_h_legs == 2, "num_h_legs should be 1 or 2!"
    if num_h_legs == 2:
        dense_bond_dim = np.sqrt(params.shape[-1])
    else:
        dense_bond_dim = params.shape[-1]

    assert h_bond_dim ** height == dense_bond_dim, f"Desired horizontal bond dimension ({h_bond_dim}) to the power of desired factored core height ({height}) must be equal to the input core's ({name}) bond dimension ({dense_bond_dim})."

    factored_params = {f"{name}_{i}":{} for i in range(height)}
    vocab_size = params.shape[0]

    ## Move two horizontal legs to left side of tensor
    remaining_core = params.reshape(vocab_size * h_bond_dim**num_h_legs, -1)
    param_type = 'embedding'

    for subcore_idx in range(height-1):
        subcore, sing_vals, remaining_core = jnp.linalg.svd(remaining_core, full_matrices=False)
        if v_bond_dim <= len(sing_vals):
            subcore = subcore[:,:v_bond_dim]
            sing_vals = sing_vals[:v_bond_dim]
            remaining_core = remaining_core[:v_bond_dim]
        else:
            warnings.warn("Large vertical bond dimension: adding noisy singular values")
            extend_len = v_bond_dim - len(sing_vals)

            ## Extend subcore (left matrix of SVD) (axis 1)
            aug_key, init_key = jax.random.split(aug_key)
            subcore_extender = jax.random.uniform(init_key, shape=(subcore.shape[0], extend_len), maxval=extend_maxval)
            subcore = jnp.concatenate((subcore, subcore_extender), axis=1)

            ## Extend singular values
            aug_key, init_key = jax.random.split(aug_key)
            extra_vals = jax.random.uniform(init_key, shape=(extend_len,), dtype=jnp.float32, minval=extend_minval, maxval=extend_maxval)
            sing_vals = jnp.concatenate((sing_vals, extra_vals))

            ## Extend remaining_core (right matrix of SVD) (axis 0)
            aug_key, init_key = jax.random.split(aug_key)
            remain_extender = jax.random.uniform(init_key, shape=(extend_len, remaining_core.shape[-1]))
            remaining_core = jnp.concatenate((remaining_core, remain_extender))

        factored_params[f"{name}_{subcore_idx}"][param_type] = subcore.reshape(-1, h_bond_dim**num_h_legs * v_bond_dim)
        param_type = 'kernel'
        remaining_core = matmul(jnp.diag(sing_vals), remaining_core)
        remaining_core = remaining_core.reshape(v_bond_dim * h_bond_dim**num_h_legs, -1)


    remaining_core = remaining_core.reshape(v_bond_dim, h_bond_dim**num_h_legs)
    factored_params[f"{name}_{height-1}"]['kernel'] = remaining_core 

    return factored_params

###########
## Cores ##
###########
class DenseCore:
    def __init__(self, num_h_legs: int, vocab_size: int, h_bond_dim: int, init_fn, name: str, dtype=jnp.float32):
        self.name = name
        self.h_shape = (h_bond_dim,) * num_h_legs

        self.bot_subcore = nn.Embed(vocab_size,
                                    (h_bond_dim**num_h_legs),
                                    name=name,
                                    embedding_init=init_fn,
                                    dtype=dtype)

    def get_params(self, params):
        return params[self.name]['embedding']

    def __call__(self, inputs):
        return self.bot_subcore(inputs).reshape(-1, *self.h_shape)


class FactoredCore(DenseCore):
    def __init__(self, num_h_legs: int, height: int, vocab_size: int, h_bond_dim: int, v_bond_dim: int, init_fn, name: str, dtype=jnp.float32):
        assert height > 1, "core height must be greater than 1!"
        self.name = name
        self.v_bond_dim = v_bond_dim
        self.h_bond_dim = h_bond_dim
        self.h_dim = h_bond_dim ** num_h_legs
        self.final_h_shape = (h_bond_dim**height,) * num_h_legs

        bot = nn.Embed(vocab_size,
                       self.h_dim * v_bond_dim,
                       name=name+"_0",
                       embedding_init=init_fn,
                       dtype=dtype)
        mid = tuple(nn.Dense(self.h_dim * v_bond_dim,
                             use_bias=False,
                             name=name+f"_{i}",
                             kernel_init=init_fn,
                             dtype=dtype)
                    for i in range(1, height-1))
        top = nn.Dense(self.h_dim,
                       use_bias=False,
                       name=name+f"_{height-1}",
                       kernel_init=init_fn,
                       dtype=dtype)
        self.subcore_tup = (bot, *mid, top)

    def v_contract(self, bot_subcore):
        core = bot_subcore.reshape(-1, self.h_dim, self.v_bond_dim)
        for i, subcore in enumerate(self.subcore_tup[1:-1], 2):
            core = subcore(core).reshape(-1, self.h_dim**i, self.v_bond_dim)
        return self.subcore_tup[-1](core).reshape(-1, *self.final_h_shape)

    def get_params(self, params):
        bot_subcore = params[f'{self.name}_0']['embedding']
        return self.v_contract(bot_subcore)

    def __call__(self, inputs):
        bot_subcore = self.subcore_tup[0](inputs)
        return self.v_contract(bot_subcore)


class SkipFactoredCore(DenseCore):
    def __init__(self, num_h_legs: int, height: int, vocab_size: int, h_bond_dim: int, v_bond_dim: int, init_fn, name: str, dtype=jnp.float32):
        assert height > 1, "core height must be greater than 1!"
        self.name = name
        self.v_bond_dim = v_bond_dim
        self.h_dim = h_bond_dim ** num_h_legs
        self.height = height
        self.final_h_shape = (h_bond_dim**height,) * num_h_legs

        bot = nn.Embed(vocab_size,
                       self.h_dim * v_bond_dim,
                       name=name+"_0",
                       embedding_init=init_fn,
                       dtype=dtype)
        mid = (nn.Embed(vocab_size,
                        self.h_dim * (v_bond_dim**2),
                        name=name+f"_{i}",
                        embedding_init=init_fn,
                        dtype=dtype)
               for i in range(1, height-1))
        top = nn.Embed(vocab_size,
                       self.h_dim * v_bond_dim,
                       name=name+f"_{height-1}",
                       embedding_init=init_fn,
                       dtype=dtype)
        self.subcore_tup = (bot, *mid, top)

    def v_contract(self, subcore_gen):
        core = next(subcore_gen).reshape(-1, self.h_dim, self.v_bond_dim)
        for i in range(2, self.height):
            ## Iterate over cores 1 to height-2, but use index + 1
            subcore = next(subcore_gen).reshape(-1, self.v_bond_dim, self.h_dim * self.v_bond_dim)
            core = matmul(core, subcore)
            core = core.reshape(-1, self.h_dim**i, self.v_bond_dim)

        top_subcore = next(subcore_gen).reshape(-1, self.v_bond_dim, self.h_dim)
        core = matmul(core, top_subcore)
        return core.reshape(-1, *self.final_h_shape)

    def get_params(self, params):
        subcore_gen = (params[f'{self.name}_{i}']['embedding'] for i in range(self.height))
        return self.v_contract(subcore_gen)

    def __call__(self, inputs):
        subcore_gen = (c(inputs) for c in self.subcore_tup)
        return self.v_contract(subcore_gen)


############
## Models ##
############
class DenseMPS(nn.Module):
    ## Assume fixed length input sequences
    num_cores: int
    vocab_size: int
    h_bond_dim: int
    embed_dtype: jnp.dtype
    partial_len: int
    boundary_var: float
    internal_var: float

    def setup(self):
        boundary_init = noisy_unit_ones_init(self.boundary_var)
        internal_init = noisy_identity_init(self.internal_var)
        left = DenseCore(1,
                         self.vocab_size,
                         self.h_bond_dim,
                         boundary_init,
                         "core_0",
                         dtype=self.embed_dtype)
        mid = (DenseCore(2,
                         self.vocab_size,
                         self.h_bond_dim,
                         internal_init,
                         f"core_{i+1}",
                         dtype=self.embed_dtype)
               for i in range(self.num_cores-2))
        right = DenseCore(1,
                          self.vocab_size,
                          self.h_bond_dim,
                          boundary_init,
                          f"core_{self.num_cores-1}",
                          dtype=self.embed_dtype)

        self.core_tup = (left, *mid, right)

    @property
    def effective_bond_dim(self):
        return self.h_bond_dim

    def __call__(self, x):
        assert x.shape[-1] == self.num_cores, f"Input shape {x.shape} not compatible with {self.num_cores}-core MPS!"

        Lcore = jnp.expand_dims(self.core_tup[0](x[:, 0]), axis=1)
        log_scalar = jnp.zeros(x.shape[0], dtype=self.embed_dtype)
        for i, core in enumerate(self.core_tup[1:-1], 1):
            contr_core = core(x[:, i])#.reshape(-1, self.effective_bond_dim, self.effective_bond_dim)
            Lcore = matmul(Lcore, contr_core)
            if i % self.partial_len == 0:
                norm = jnp.linalg.norm(Lcore, axis=-1)
                log_scalar += jnp.log(norm).flatten()
                Lcore = Lcore / norm[:, None]

        Rcore = jnp.expand_dims(self.core_tup[-1](x[:, -1]), axis=2)
        Lcore = matmul(Lcore, Rcore)
        norm = jnp.linalg.norm(Lcore, axis=-1)
        log_scalar += jnp.log(norm).flatten()

        return log_scalar


    ## Normalization functions
    def get_core_list(self, params):
        return (c.get_params(params) for c in self.core_tup)

    def contract_TB(self, Tcore, Bcore, lns):
        cap = matmul(Bcore.T, Tcore)
        cap_norm = jnp.linalg.norm(cap)
        cap /= cap_norm
        lns += jnp.log(cap_norm)
        return cap, lns

    def LNS(self, params):
        """
        core_list should be of the shape [(V, chi), (V, chi^2), ..., (V, chi^2), (d, chi)]
        where V is vocab_size and chi is the effective bond_dim
        """
        core_map = self.apply({'params': params}, params, method=self.get_core_list)

        core = next(core_map)
        cap, lns = self.contract_TB(core, core, 0)

        for _ in range(self.num_cores-2):
            core = next(core_map)
            Tcore = matmul(cap, core.reshape(self.vocab_size, self.effective_bond_dim, self.effective_bond_dim)).reshape(-1, self.effective_bond_dim)
            Bcore = core.reshape(-1, self.effective_bond_dim)
            cap, lns = self.contract_TB(Tcore, Bcore, lns)

        core = next(core_map)
        Tcore = matmul(core, cap.T).reshape(-1)
        Bcore = core.reshape(-1)
        lns += jnp.log(matmul(Bcore, Tcore))

        return lns

    def param_norm(self, params):
        core_map = self.apply({'params': params}, params, method=self.get_core_list)

        norm = jnp.square(next(core_map)).sum()

        for _ in range(self.num_cores-2):
            core = next(core_map)            
            iden = jnp.eye(self.h_bond_dim).flatten()
            norm += jnp.square(core - iden).sum()

        norm += jnp.square(next(core_map)).sum()
        return norm 


class FactoredMPS(DenseMPS):
    ## Assume fixed length input sequences
    core_height: int
    v_bond_dim: int
    skip_connection: bool

    def setup(self):
        boundary_init = noisy_zeros_init(self.boundary_var)
        internal_init = noisy_zeros_init(self.internal_var)
        core_params = (self.core_height, self.vocab_size, self.h_bond_dim, self.v_bond_dim)
        if self.skip_connection:
            Core = SkipFactoredCore
        else:
            Core = FactoredCore

        left = Core(1,
                    *core_params,
                    boundary_init,
                    name="core_0",
                    dtype=self.embed_dtype)
        mid = (Core(2,
                    *core_params,
                    internal_init,
                    name=f'core_{i+1}',
                    dtype=self.embed_dtype)
               for i in range(self.num_cores-2))
        right = Core(1,
                     *core_params,
                     boundary_init,
                     name=f"core_{self.num_cores-1}",
                     dtype=self.embed_dtype)
        self.core_tup = (left, *mid, right)

    @property
    def effective_bond_dim(self):
        return self.h_bond_dim ** self.core_height


##############
## Sampling ##
##############
def get_next_cap(vocab_size, bond_dim, precision):
    def next_cap(Tcore, Bcore, cap):
        CTcore = jnp.matmul(Tcore.reshape(vocab_size, bond_dim, bond_dim), cap, precision=precision)
        CTcore = CTcore.transpose(1, 0, 2).reshape(bond_dim, -1)

        Bcore = Bcore.reshape(vocab_size, bond_dim, bond_dim)
        Bcore = Bcore.transpose(0, 2, 1).reshape(-1, bond_dim)

        cap = jnp.matmul(CTcore, Bcore, precision=precision).reshape(1, bond_dim, bond_dim)
        cap_norm = jnp.linalg.norm(cap)
        cap /= cap_norm
        return cap
    return next_cap


def get_sampler(model, bond_dim, precision=jax.lax.Precision.HIGHEST):
    ## set up the function to sample model
    ## bond dim is the effective bond dim of the dense core
    n_cores = model.num_cores
    vocab_size = model.vocab_size

    matmul_samp = functools.partial(jnp.matmul, precision=precision)
    contract_TB = get_contract_TB(bond_dim, precision)

    def pre_sampler(params):
        """
        Cores have shape (vocab_size, bond_dim**2),
        except first/last, which have shape (vocab_size, bond_dim)
        """

        core_list = model.apply({'params': params}, params, method=model.get_core_list)

        """
        Generate list of right caps (in reverse) and their log_norms
        """
        core = core_list[-1]
        cap, _ = contract_TB(core, core, 0)
        caps_rev = [cap]

        for cap_i in range(1, n_cores-1):
            core = core_list[-1-cap_i].reshape(vocab_size, bond_dim, bond_dim)
            CTcore = matmul_samp(core, caps_rev[-1])
            CTcore = CTcore.transpose(1, 0, 2).reshape(bond_dim, -1)

            Bcore = core.transpose(0, 2, 1).reshape(-1, bond_dim)

            cap = matmul_samp(CTcore, Bcore).reshape(1, bond_dim, bond_dim)
            cap_norm = jnp.linalg.norm(cap)
            cap /= cap_norm
            caps_rev.append(cap)

        return (caps_rev, core_list)

    def sampler(caps_rev, core_list, key):
        """
        find marginal distribution for site i, conditioned on sites 0 - i-1
        """
        core = core_list[0]
        CTcore = matmul_samp(core, caps_rev[-1])

        pre_density = matmul_samp(CTcore.reshape(-1, bond_dim), core.T)
        ## Normalize using the trace
        density = pre_density / jnp.trace(pre_density)
        key, samp_key = random.split(key)
        tok_list = [random.choice(samp_key, vocab_size, p=density.diagonal()).item()]

        Lcore = core_list[0][tok_list[0]]

        for site_i in range(1, n_cores-1):
            midcore = core_list[site_i].reshape(vocab_size, bond_dim, bond_dim)

            core = matmul_samp(Lcore, midcore)

            cap = caps_rev[-(site_i+1)].reshape(bond_dim, bond_dim)
            Ccore = matmul_samp(core, cap)
            pre_density = matmul_samp(Ccore, core.T)
            ## Normalize using the trace
            density = pre_density / jnp.trace(pre_density)

            key, samp_key = random.split(key)
            tok_list.append(random.choice(samp_key, vocab_size, p=density.diagonal()).item())

            Lcore = matmul_samp(Lcore, core_list[site_i][tok_list[-1]].reshape(bond_dim, bond_dim))

        fin = matmul_samp(Lcore, core_list[-1].T)
        pre_density = jnp.outer(fin, fin)
        density = pre_density / jnp.trace(pre_density)

        key, samp_key = random.split(key)
        tok_list.append(random.choice(samp_key, vocab_size, p=density.diagonal()).item())
        del samp_key

        return tok_list

    return pre_sampler, sampler


def get_sample_generator(model, bond_dim, precision=jax.lax.Precision.HIGHEST):
    pre_sampler, sampler = get_sampler(model, bond_dim, precision)

    def sample_generator(params, key, num_samples):
        sampler_args = pre_sampler(params)

        for step in range(num_samples):
            key, samp_key = random.split(key)
            yield step, sampler(*sampler_args, samp_key)

    return sample_generator
