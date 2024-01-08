"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import jax.numpy as jnp
from flax import linen as nn


class SLP(nn.Module):
    vocab_size: int
    embed_dim: int
    embed_dtype: jnp.dtype
    hidden_layer_size: int

    def setup(self):
        self.embed = nn.Embed(self.vocab_size,
                              self.embed_dim,
                              name="embed",
                              dtype=self.embed_dtype)
        self.hidden = nn.Dense(self.hidden_layer_size,
                               dtype=self.embed_dtype,
                               name="hidden")
        self.output = nn.Dense(1,
                               dtype=self.embed_dtype,
                               name="output")

    def __call__(self, x):
        ## Model outputs log(prob(x))
        x = self.embed(x)
        x = x.reshape(len(x),-1)
        x = self.hidden(x)
        x = nn.relu(x)
        x = self.output(x)
        return nn.log_sigmoid(x).flatten()

    def LNS(self, params):
        return 0.0
