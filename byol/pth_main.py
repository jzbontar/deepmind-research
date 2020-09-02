import unittest
import pickle

import torch
from torch import nn
import torchvision

import haiku as hk
from jax import random
import jax.numpy as jnp

import numpy as np

from byol.utils import networks
import byol.jzb_resnet

def j2t(x, cuda=True):
    y = torch.from_numpy(np.asarray(x).copy())
    if cuda:
        y = y.cuda()
    return y

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
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            return networks.MLP('predictor', 8, 4, bn_config)(inputs, is_training=is_training)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (4, 3))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        sd = sd_j2t(convert_mlp(params, state))
        m = MLP(3, 8, 4).cuda()
        m.load_state_dict(sd)
        tout = m(j2t(x))

        assert allclose(jout, tout)

    def test_resnet(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            return networks.ResNet18(bn_config=bn_config)(inputs, is_training)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (4, 32, 32, 3))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        # pickle.dump((params, state), open('foo.pkl', 'wb'))

        params, state = pickle.load(open('foo.pkl', 'rb'))
        sd = sd_j2t(convert_resnet(params, state))
        m = byol.jzb_resnet.resnet18().cuda()
        m.load_state_dict(sd)
        tout = m(j2t(x).permute(0, 3, 1, 2))

        print((j2t(jout) - tout).abs())

    def test_conv(self):
        def _forward(inputs, is_training):
            return hk.Conv2D(output_channels=8, kernel_shape=3, stride=1, with_bias=False, padding='SAME', name='conv')(inputs)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (5, 32, 32, 3))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        m = nn.Conv2d(3, 8, 3, 1, 1, bias=False).cuda()
        m.load_state_dict(sd_j2t(convert_conv('conv', params)))
        tout = m(j2t(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        assert allclose(jout, tout, atol=1e-5)

def convert_resnet(params, state):
    n = list(params.keys())[0].split('/')[0] + '/~'

    return dict(
        **add_prefix('conv1', convert_conv(f'{n}/initial_conv', params)),
        **add_prefix('bn1', convert_batchnorm(f'{n}/initial_batchnorm', params, state)),
        **add_prefix('layer1', convert_blockgroup(f'{n}/block_group_0/~', params, state)),
        **add_prefix('layer2', convert_blockgroup(f'{n}/block_group_1/~', params, state)),
        **add_prefix('layer3', convert_blockgroup(f'{n}/block_group_2/~', params, state)),
        **add_prefix('layer4', convert_blockgroup(f'{n}/block_group_3/~', params, state)),
    )

def convert_blockgroup(prefix, params, state):
    return dict(
        **add_prefix('0', convert_block(f'{prefix}/block_0/~', params, state)),
        **add_prefix('1', convert_block(f'{prefix}/block_1/~', params, state)),
    )

def convert_block(prefix, params, state):
    d = dict(
        **add_prefix('conv1', convert_conv(f'{prefix}/conv_0', params)),
        **add_prefix('bn1', convert_batchnorm(f'{prefix}/batchnorm_0', params, state)),
        **add_prefix('conv2', convert_conv(f'{prefix}/conv_1', params)),
        **add_prefix('bn2', convert_batchnorm(f'{prefix}/batchnorm_1', params, state)),
    )
    if prefix.endswith('/block_0/~'):
        d.update(
            **add_prefix('downsample.0', convert_conv(f'{prefix}/shortcut_conv', params)),
            **add_prefix('downsample.1', convert_batchnorm(f'{prefix}/shortcut_batchnorm', params, state)),
        )
    return d
    
def convert_conv(prefix, params):
    return dict(weight=params[prefix]['w'].transpose((3, 2, 0, 1)), bias=params[prefix].get('b'))

def add_prefix(prefix, params):
    return {f'{prefix}.{k}': v for k, v in params.items()}

def sd_j2t(sd):
    return {k: j2t(v, False) for k, v in sd.items() if v is not None}

def convert_mlp(params, state):
    assert len(params.keys()) == 3
    n = list(params.keys())[0].split('/')[0]
    return dict(
        **add_prefix('0', convert_linear(f'{n}/linear', params)),
        **add_prefix('1', convert_batchnorm(f'{n}/batch_norm', params, state)),
        **add_prefix('3', convert_linear(f'{n}/linear_1', params)))

def convert_batchnorm(prefix, params, state):
    return dict(
        weight=params[prefix]['scale'].ravel(), 
        bias=params[prefix]['offset'].ravel(), 
        running_mean=state[f'{prefix}/~/mean_ema']['hidden'].ravel(), 
        running_var=state[f'{prefix}/~/var_ema']['hidden'].ravel())

def convert_linear(prefix, params):
    return dict(weight=params[prefix]['w'].T, bias=params[prefix].get('b'))

def MLP(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=True),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size, bias=False))

