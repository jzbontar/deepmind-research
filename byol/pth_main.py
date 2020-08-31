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

def allclose(jx, tx, **kwargs):
    return torch.allclose(j2t(jx), tx, **kwargs)

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

    def test_bn(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            return hk.BatchNorm(**bn_config)(inputs, is_training=is_training)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = [random.normal(kk, (4, 3)) for kk in random.split(k, 4)]

        params, state0 = forward.init(k, x[0], True)
        jout1, state1 = forward.apply(params, state0, x[1], True)
        jout2, state2 = forward.apply(params, state1, x[2], True)
        jout3, state3 = forward.apply(params, state2, x[3], False)

        m = nn.BatchNorm1d(3).cuda()
        m.running_var.zero_()
        tout1 = m.forward(j2t(x[1]))
        tout2 = m.forward(j2t(x[2]))

        def zero_debias(x, decay, counter):
            return x / (1 - decay**counter)

        batch_size = 4
        m.running_mean = zero_debias(m.running_mean, 1 - m.momentum, m.num_batches_tracked)
        m.running_var = zero_debias(m.running_var * (batch_size - 1) / batch_size, 1 - m.momentum, m.num_batches_tracked)
        tout3 = m.eval().forward(j2t(x[3]))

        assert allclose(jout1, tout1)
        assert allclose(jout2, tout2)
        assert allclose(jout3, tout3, atol=1e-7)

    def test_mlp(self):
        pass
