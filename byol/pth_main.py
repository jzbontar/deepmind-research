import unittest

import torch
from torch import nn

import haiku as hk
from jax import random
import jax.numpy as jnp

import numpy as np

from byol.utils import networks

def j2t(x):
    return torch.from_numpy(np.asarray(x).copy()).cuda()

def allclose(jx, tx):
    return torch.allclose(j2t(jx), tx)

class TestBYOL(unittest.TestCase):
    def test_linear(self):
        def _forward(inputs):
            return hk.Linear(output_size=4, with_bias=True)(inputs)

        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        inputs = random.normal(k, (2, 3))
        params, state = forward.init(k, inputs)
        jout, new_state = forward.apply(params, state, inputs)

        m = nn.Linear(3, 4).cuda()
        sd = {}
        sd['weight'] = j2t(params['linear']['w'].T)
        sd['bias'] = j2t(params['linear']['b'])
        m.load_state_dict(sd)
        tout = m(j2t(inputs))

        assert allclose(jout, tout)
