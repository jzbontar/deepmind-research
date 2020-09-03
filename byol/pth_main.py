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

def j2t(x):
    y = torch.from_numpy(np.asarray(x).copy()).cuda()
    if y.ndim == 4:
        y = y.permute(0, 3, 1, 2)
    return y

def allclose(jx, tx, **kwargs):
    close = torch.allclose(j2t(jx), tx, **kwargs)
    if not close:
        print((j2t(jx) - tx).abs().max())
    return close
        

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

        sd = sd_j2t(convert_mlp('predictor', params, state))
        m = MLP(3, 8, 4).cuda()
        m.load_state_dict(sd)
        tout = m(j2t(x))

        assert allclose(jout, tout, atol=1e-7)

    def test_resnet(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            return networks.ResNet18(bn_config=bn_config)(inputs, is_training)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (4, 32, 32, 3))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        sd = sd_j2t(convert_resnet('res_net18/~', params, state))
        m = byol.jzb_resnet.resnet18().cuda()
        m.load_state_dict(sd)
        tout = m(j2t(x))

        assert allclose(jout, tout, atol=1e-3)

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
        tout = m(j2t(x))

        assert allclose(jout, tout, atol=1e-5)

    def test_blockgroup(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            resnet = networks.ResNet18(bn_config=bn_config)
            m = resnet.block_groups[3]
            return m(inputs, is_training, False)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (2, 14, 14, 256))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        resnet = byol.jzb_resnet.resnet18().cuda()
        m = resnet.layer4
        m.load_state_dict(sd_j2t(convert_blockgroup('res_net18/~/block_group_3/~', params, state)))
        tout = m(j2t(x))

        assert allclose(jout, tout, atol=1e-5)

    def test_block(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            resnet = networks.ResNet18(bn_config=bn_config)
            m = resnet.block_groups[3].blocks[0]
            return m(inputs, is_training, False)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (2, 14, 14, 256))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        resnet = byol.jzb_resnet.resnet18().cuda()
        m = resnet.layer4[0]
        m.load_state_dict(sd_j2t(convert_block('res_net18/~/block_group_3/~/block_0/~', params, state)))
        tout = m(j2t(x))

        assert allclose(jout, tout, atol=1e-5)

    def test_maxpool(self):
        def _forward(inputs):
            return hk.max_pool(inputs, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (2, 112, 112, 64))
        params, state = forward.init(k, x)
        jout, _ = forward.apply(params, state, x)
        
        m = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), -2e38),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        tout = m(j2t(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        assert allclose(jout, tout, atol=1e-5)

    def test_byol_forward(self):
        import byol.byol_experiment

        k0, k1, k2, k3 = random.split(random.PRNGKey(0), 4)
        input = dict(
            view1=random.normal(k1, (2, 128, 128, 3)),
            view2=random.normal(k2, (2, 128, 128, 3)),
            labels=random.randint(k3, (2,), 0, 9))

        kwargs = {'random_seed': 0, 'num_classes': 10, 'batch_size': 256, 'max_steps': 36988, 'enable_double_transpose': True, 'base_target_ema': 0.996, 'network_config': {'projector_hidden_size': 4096, 'projector_output_size': 256, 'predictor_hidden_size': 4096, 'encoder_class': 'ResNet18', 'encoder_config': {'resnet_v2': False, 'width_multiplier': 1}, 'bn_config': {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}}, 'optimizer_config': {'weight_decay': 1e-06, 'eta': 0.001, 'momentum': 0.9}, 'lr_schedule_config': {'base_learning_rate': 2.0, 'warmup_steps': 369}, 'evaluation_config': {'subset': 'test', 'batch_size': 25}, 'checkpointing_config': {'use_checkpointing': True, 'checkpoint_dir': '/scratch/jzb/byol_checkpoints', 'save_checkpoint_interval': 300, 'filename': 'pretrain.pkl'}}
        exp = byol.byol_experiment.ByolExperiment(**kwargs)
        params, state = exp.forward.init(k0, input, is_training=True)
        jout, _ = exp.forward.apply(params, state, input, is_training=True)
        
        m = ByolModel(10, 512, 4096, 256, 4096).cuda()
        sd = sd_j2t(dict(
            **add_prefix('classifier', convert_linear('classifier', params)),
            **add_prefix('predictor', convert_mlp('predictor', params, state)),
            **add_prefix('projector', convert_mlp('projector', params, state)),
            **add_prefix('net', convert_resnet('res_net18/~', params, state)),
        ))
        m.load_state_dict(sd)
        tout = m({k: j2t(v) for k, v in input.items()})

        assert jout.keys() == tout.keys()
        for k in jout.keys():
            assert allclose(jout[k], tout[k], atol=2e-3)


def normalize_images(images):
    """Normalize the image using ImageNet statistics."""
    mean_rgb = (0.485, 0.456, 0.406)
    stddev_rgb = (0.229, 0.224, 0.225)
    normed_images = images - torch.Tensor(mean_rgb).view(1, 3, 1, 1).cuda()
    normed_images = normed_images / torch.Tensor(stddev_rgb).view(1, 3, 1, 1).cuda()
    return normed_images

class ByolModel(nn.Module):
    def __init__(self, num_classes, projector_input_size, projector_hidden_size, projector_output_size, predictor_hidden_size):
        super().__init__()
        self.projector = MLP(projector_input_size, projector_hidden_size, projector_output_size)
        self.predictor = MLP(projector_output_size, predictor_hidden_size, projector_output_size)
        self.classifier = nn.Linear(projector_input_size, num_classes)
        self.net = byol.jzb_resnet.resnet18()

    def forward(self, inputs):
        def apply_once_fn(images, suffix):
            images = normalize_images(images)

            embedding = self.net(images)
            proj_out = self.projector(embedding)
            pred_out = self.predictor(proj_out)

            # Note the stop_gradient: label information is not leaked into the
            # main network.
            classif_out = self.classifier(embedding.detach())
            outputs = {}
            outputs['projection' + suffix] = proj_out
            outputs['prediction' + suffix] = pred_out
            outputs['logits' + suffix] = classif_out
            return outputs

        if self.training:
            outputs_view1 = apply_once_fn(inputs['view1'], '_view1')
            outputs_view2 = apply_once_fn(inputs['view2'], '_view2')
            return {**outputs_view1, **outputs_view2}
        else:
            return apply_once_fn(inputs['images'], '')

def convert_resnet(prefix, params, state):
    return dict(
        **add_prefix('conv1', convert_conv(f'{prefix}/initial_conv', params)),
        **add_prefix('bn1', convert_batchnorm(f'{prefix}/initial_batchnorm', params, state)),
        **add_prefix('layer1', convert_blockgroup(f'{prefix}/block_group_0/~', params, state)),
        **add_prefix('layer2', convert_blockgroup(f'{prefix}/block_group_1/~', params, state)),
        **add_prefix('layer3', convert_blockgroup(f'{prefix}/block_group_2/~', params, state)),
        **add_prefix('layer4', convert_blockgroup(f'{prefix}/block_group_3/~', params, state)),
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
    return {k: torch.from_numpy(np.asarray(v).copy()) for k, v in sd.items() if v is not None}

def convert_mlp(prefix, params, state):
    return dict(
        **add_prefix('0', convert_linear(f'{prefix}/linear', params)),
        **add_prefix('1', convert_batchnorm(f'{prefix}/batch_norm', params, state)),
        **add_prefix('3', convert_linear(f'{prefix}/linear_1', params)))

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

