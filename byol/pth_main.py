import unittest
import pickle

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

import jax
import haiku as hk
from jax import random
import jax.numpy as jnp
import optax

import numpy as np

from byol.utils import augmentations
from byol.utils import networks
from byol.utils import dataset
import byol.byol_experiment
import byol.jzb_resnet

def j2p_tensor(x):
    y = torch.from_numpy(np.asarray(x).copy())
    if y.dtype == torch.int32:
        y = y.long()
    if y.ndim == 4:
        y = y.permute(0, 3, 1, 2)
    y = y.cuda()
    return y

def allclose(jx, tx, **kwargs):
    close = torch.allclose(j2p_tensor(jx), tx, **kwargs)
    if not close:
        print((j2p_tensor(jx) - tx).abs().max())
    return close
        

class TestBYOL(unittest.TestCase):
    def test_linear(self):
        def _forward(inputs):
            return hk.Linear(output_size=4, with_bias=True)(inputs)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        inputs = random.normal(k, (2, 3))
        params, state = forward.init(k, inputs)
        jout, _ = forward.apply(params, state, inputs)

        m = nn.Linear(3, 4).cuda()
        sd = {}
        sd['weight'] = j2p_tensor(params['linear']['w'].T)
        sd['bias'] = j2p_tensor(params['linear']['b'])
        m.load_state_dict(sd)
        tout = m(j2p_tensor(inputs))
        assert allclose(jout, tout)

        params2 = p2j_linear(m, 'linear')
        jout2, _ = forward.apply(params2, state, inputs)
        assert allclose(jout2, tout)


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

        m = nn.BatchNorm1d(3).cuda()
        m.load_state_dict(j2p_sd(j2p_batchnorm('batch_norm', params, state0)))
        tout1 = m.forward(j2p_tensor(x[1]))
        tout2 = m.forward(j2p_tensor(x[2]))
        assert allclose(jout1, tout1)
        assert allclose(jout2, tout2)

        jout3, state3 = forward.apply(params, state2, x[3], False)
        params4, state4 = p2j_batchnorm(m, 'batch_norm')
        jout4, _ = forward.apply(params4, state4, x[3], False)
        assert allclose(jout3, j2p_tensor(jout4), atol=1e-7)

    def test_mlp(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            return networks.MLP('predictor', 8, 4, bn_config)(inputs, is_training=is_training)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (4, 3))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        sd = j2p_sd(j2p_mlp('predictor', params, state))
        m = MLP(3, 8, 4).cuda()
        m.load_state_dict(sd)
        tout = m(j2p_tensor(x))
        assert allclose(jout, tout, atol=1e-7)

        params1, state1 = p2j_mlp(m, 'predictor')
        jout1, _ = forward.apply(params1, state1, x, True)
        assert allclose(jout, j2p_tensor(jout1))

    def test_resnet(self):
        def _forward(inputs, is_training):
            bn_config = {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}
            return networks.ResNet18(bn_config=bn_config)(inputs, is_training)
        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        x = random.normal(k, (4, 32, 32, 3))
        params, state = forward.init(k, x, True)
        jout, _ = forward.apply(params, state, x, True)

        sd = j2p_sd(j2p_resnet('res_net18/~', params, state))
        m = byol.jzb_resnet.resnet18().cuda()
        m.load_state_dict(sd)
        tout = m(j2p_tensor(x))

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
        m.load_state_dict(j2p_sd(j2p_conv('conv', params)))
        tout = m(j2p_tensor(x))
        assert allclose(jout, tout, atol=1e-5)

        params1 = p2j_conv(m, 'conv')
        jout1, _ = forward.apply(params1, state, x, True)
        assert allclose(jout, j2p_tensor(jout1))


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
        m.load_state_dict(j2p_sd(j2p_blockgroup('res_net18/~/block_group_3/~', params, state)))
        tout = m(j2p_tensor(x))

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
        m.load_state_dict(j2p_sd(j2p_block('res_net18/~/block_group_3/~/block_0/~', params, state)))
        tout = m(j2p_tensor(x))
        assert allclose(jout, tout, atol=1e-5)

        params1, state1 = p2j_block(m, 'res_net18/~/block_group_3/~/block_0/~')
        jout1, _ = forward.apply(params1, state1, x, True)
        assert allclose(jout, j2p_tensor(jout1))

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
        tout = m(j2p_tensor(x))

        assert allclose(jout, tout, atol=1e-5)

    def test_byol_forward(self):
        k0, k1, k2, k3 = random.split(random.PRNGKey(0), 4)
        input = dict(
            view1=random.normal(k1, (2, 128, 128, 3)),
            view2=random.normal(k2, (2, 128, 128, 3)),
            labels=random.randint(k3, (2,), 0, 9))

        kwargs = {'random_seed': 0, 'num_classes': 10, 'batch_size': 256, 'max_steps': 36988, 'enable_double_transpose': True, 'base_target_ema': 0.996, 'network_config': {'projector_hidden_size': 4096, 'projector_output_size': 256, 'predictor_hidden_size': 4096, 'encoder_class': 'ResNet18', 'encoder_config': {'resnet_v2': False, 'width_multiplier': 1}, 'bn_config': {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}}, 'optimizer_config': {'weight_decay': 1e-06, 'eta': 0.001, 'momentum': 0.9}, 'lr_schedule_config': {'base_learning_rate': 2.0, 'warmup_steps': 369}, 'evaluation_config': {'subset': 'test', 'batch_size': 25}, 'checkpointing_config': {'use_checkpointing': True, 'checkpoint_dir': '/scratch/jzb/byol_checkpoints', 'save_checkpoint_interval': 300, 'filename': 'pretrain.pkl'}}
        exp = byol.byol_experiment.ByolExperiment(**kwargs)
        params, state = exp.forward.init(k0, input, is_training=True)
        jout, _ = exp.forward.apply(params, state, input, is_training=True)
        # pickle.dump((params, state), open('foo.pkl', 'wb'))
        
        m = ByolNetwork().cuda()
        sd = j2p_sd(j2p_byol_network(params, state))
        m.load_state_dict(sd)
        tout = m({k: j2p_tensor(v) for k, v in input.items()})

        assert jout.keys() == tout.keys()
        for k in jout.keys():
            assert allclose(jout[k], tout[k], atol=2e-3)
    
    def test_loss_fn(self):
        k0, k1, k2, k3 = random.split(random.PRNGKey(0), 4)
        batch_size = 4
        input = dict(
            view1=random.normal(k1, (batch_size, 128, 128, 3)),
            view2=random.normal(k2, (batch_size, 128, 128, 3)),
            labels=random.randint(k3, (batch_size,), 0, 9))

        kwargs = {'random_seed': 0, 'num_classes': 10, 'batch_size': batch_size, 'max_steps': 36988, 'enable_double_transpose': True, 'base_target_ema': 0.996, 'network_config': {'projector_hidden_size': 4096, 'projector_output_size': 256, 'predictor_hidden_size': 4096, 'encoder_class': 'ResNet18', 'encoder_config': {'resnet_v2': False, 'width_multiplier': 1}, 'bn_config': {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}}, 'optimizer_config': {'weight_decay': 1e-06, 'eta': 0.001, 'momentum': 0.9}, 'lr_schedule_config': {'base_learning_rate': 2.0, 'warmup_steps': 369}, 'evaluation_config': {'subset': 'test', 'batch_size': 25}, 'checkpointing_config': {'use_checkpointing': True, 'checkpoint_dir': '/scratch/jzb/byol_checkpoints', 'save_checkpoint_interval': 300, 'filename': 'pretrain.pkl'}}
        exp = byol.byol_experiment.ByolExperiment(**kwargs)
        state = exp._make_initial_state(k0, input)
        jout = exp.loss_fn(rng=k0, inputs=input, 
            online_params=state.online_params, 
            online_state=state.online_state, 
            target_params=state.target_params, 
            target_state=state.target_state)

        m = ByolModule().cuda()
        sd = j2p_sd(j2p_byol_module(state))
        m.load_state_dict(sd)
        tout = m(input, k0)

        jout = jout[1][1]
        tout = tout[1]

        assert jout.keys() == tout.keys()
        for k in jout.keys():
            assert allclose(jout[k], tout[k])

    def test_scale(self):
        k = random.split(random.PRNGKey(0), 2)
        params = [random.normal(k[0], (4,))]
        grads = [random.normal(k[1], (4,))]

        t = optax.scale(2)
        state = t.init(params)
        jgrads, state = t.update(grads, state, params)

        tparams = []
        for p, g in zip(params, grads):
            tparam = j2p_tensor(p)
            tparam.grad = j2p_tensor(g)
            tparams.append(tparam)

        t = Scale(2)
        t.init(tparams)
        t.update()
        tgrads = [p.grad for p in tparams]

        assert allclose(jgrads[0], tgrads[0])

def j2p_batchnorm(prefix, params, state):
    assert state[f'{prefix}/~/var_ema']['counter'] == 0
    return dict(
        weight=params[prefix]['scale'].ravel(), 
        bias=params[prefix]['offset'].ravel(), 
        running_mean=state[f'{prefix}/~/mean_ema']['hidden'].ravel(), 
        running_var=state[f'{prefix}/~/var_ema']['hidden'].ravel(),
        num_batches_tracked=0)

def p2j_batchnorm(m, prefix, ndim=2):
    if ndim == 2:
        w = m.weight[None]
        b = m.bias[None]
    elif ndim == 4:
        w = m.weight[None, None, None]
        b = m.bias[None, None, None]
    params = {prefix: dict(scale=p2j_tensor(w), offset=p2j_tensor(b))}
    def ema(x):
        return dict(
            counter=p2j_tensor(m.num_batches_tracked),
            hidden=p2j_tensor(x),
            average=p2j_tensor(x / (1 - 0.9**m.num_batches_tracked)))
    state = {
        f'{prefix}/~/mean_ema': ema(m.running_mean),
        f'{prefix}/~/var_ema': ema(3 / 4 * m.running_var),
    }
    return params, state

def p2j_tensor(x):
    return jnp.array(x.detach().cpu().numpy())

def p2j_byol_module(m):
    return dict(target=p2j_byol_network(root.target), online=p2j_byol_network(root.online))

def p2j_byol_network(m):
    return dict(
        **p2j_mlp(root.projector, f'{prefix}projector/'),
        **p2j_mlp(root.predictor, f'{prefix}predictor/'),
        **p2j_linear(root.classifier, f'{prefix}classifier/'),
        **p2j_resnet(root.net, f'{prefix}res_net18/~/'))

def p2j_linear(m, prefix):
    params = dict(w=p2j_tensor(m.weight.T))
    if m.bias is not None:
        params['b'] = p2j_tensor(m.bias)
    return {prefix: params}

def p2j_mlp(m, prefix):
    bn_params, bn_state = p2j_batchnorm(m[1], f'{prefix}/batch_norm')
    params = {
        **p2j_linear(m[0], f'{prefix}/linear'),
        **bn_params,
        **p2j_linear(m[3], f'{prefix}/linear_1')}
    return params, bn_state

def p2j_resnet(m, prefix):
    return {}
        
class AddWeightDecay:
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay

    def init(self, params):
        self.params = params

    def update(self):
        for p in self.params:
            if p.ndim == 1:
                continue
            p.grad.add_(p, alpha=self.weight_decay)

class Scale:
    def __init__(self, scale_coeff):
        self.scale_coeff = scale_coeff

    def init(self, params):
        self.params = params

    def update(self):
        for p in self.params:
            p.grad.mul_(self.scale_coeff)


def j2p_byol_module(state):
    return dict(
        **add_prefix('online', j2p_byol_network(state.online_params, state.online_state)),
        **add_prefix('target', j2p_byol_network(state.target_params, state.target_state)),
    )
    
def normalize_images(images):
    """Normalize the image using ImageNet statistics."""
    mean_rgb = (0.485, 0.456, 0.406)
    stddev_rgb = (0.229, 0.224, 0.225)
    normed_images = images - torch.Tensor(mean_rgb).view(1, 3, 1, 1).cuda()
    normed_images = normed_images / torch.Tensor(stddev_rgb).view(1, 3, 1, 1).cuda()
    return normed_images

def regression_loss(x, y):
    normed_x, normed_y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    return torch.sum((normed_x - normed_y).pow(2), dim=1)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res

class ByolModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.online = ByolNetwork()
        self.target = ByolNetwork()

    def forward(self, inputs, rng):
        inputs = augmentations.postprocess(inputs, rng)
        inputs = {k: j2p_tensor(v) for k, v in inputs.items()}
        labels = inputs['labels']

        online_network_out = self.online(inputs)
        target_network_out = self.target(inputs)

        # Representation loss

        repr_loss = regression_loss(
            online_network_out['prediction_view1'],
            target_network_out['projection_view2'].detach())

        repr_loss = repr_loss + regression_loss(
            online_network_out['prediction_view2'],
            target_network_out['projection_view1'].detach())

        repr_loss = torch.mean(repr_loss)

        # Classification loss (with gradient flows stopped from flowing into the
        # ResNet). This is used to provide an evaluation of the representation
        # quality during training.

        classif_loss = F.cross_entropy(online_network_out['logits_view1'], labels)

        acc1, acc5 = accuracy(online_network_out['logits_view1'], labels, topk=(1, 5))

        loss = repr_loss + classif_loss
        logs = dict(
            loss=loss,
            repr_loss=repr_loss,
            classif_loss=classif_loss,
            top1_accuracy=acc1,
            top5_accuracy=acc5,
        )

        return loss, logs


class ByolNetwork(nn.Module):
    def __init__(self, num_classes=10, projector_input_size=512, projector_hidden_size=4096, projector_output_size=256, predictor_hidden_size=4096):
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

def j2p_byol_network(params, state):
    return dict(
        **add_prefix('classifier', j2p_linear('classifier', params)),
        **add_prefix('predictor', j2p_mlp('predictor', params, state)),
        **add_prefix('projector', j2p_mlp('projector', params, state)),
        **add_prefix('net', j2p_resnet('res_net18/~', params, state)),
    )

def j2p_resnet(prefix, params, state):
    return dict(
        **add_prefix('conv1', j2p_conv(f'{prefix}/initial_conv', params)),
        **add_prefix('bn1', j2p_batchnorm(f'{prefix}/initial_batchnorm', params, state)),
        **add_prefix('layer1', j2p_blockgroup(f'{prefix}/block_group_0/~', params, state)),
        **add_prefix('layer2', j2p_blockgroup(f'{prefix}/block_group_1/~', params, state)),
        **add_prefix('layer3', j2p_blockgroup(f'{prefix}/block_group_2/~', params, state)),
        **add_prefix('layer4', j2p_blockgroup(f'{prefix}/block_group_3/~', params, state)),
    )

def j2p_blockgroup(prefix, params, state):
    return dict(
        **add_prefix('0', j2p_block(f'{prefix}/block_0/~', params, state)),
        **add_prefix('1', j2p_block(f'{prefix}/block_1/~', params, state)),
    )

def j2p_block(prefix, params, state):
    d = dict(
        **add_prefix('conv1', j2p_conv(f'{prefix}/conv_0', params)),
        **add_prefix('bn1', j2p_batchnorm(f'{prefix}/batchnorm_0', params, state)),
        **add_prefix('conv2', j2p_conv(f'{prefix}/conv_1', params)),
        **add_prefix('bn2', j2p_batchnorm(f'{prefix}/batchnorm_1', params, state)),
    )
    if prefix.endswith('/block_0/~'):
        d.update(
            **add_prefix('downsample.0', j2p_conv(f'{prefix}/shortcut_conv', params)),
            **add_prefix('downsample.1', j2p_batchnorm(f'{prefix}/shortcut_batchnorm', params, state)),
        )
    return d

def p2j_block(m, prefix):
    bn1_params, bn1_state = p2j_batchnorm(m.bn1, f'{prefix}/batchnorm_0', ndim=4)
    bn2_params, bn2_state = p2j_batchnorm(m.bn2, f'{prefix}/batchnorm_1', ndim=4)
    c1_params = p2j_conv(m.conv1, f'{prefix}/conv_0')
    c2_params = p2j_conv(m.conv2, f'{prefix}/conv_1')
    params = dict(**bn1_params, **bn2_params, **c1_params, **c2_params)
    state = dict(**bn1_state, **bn2_state)
    if hasattr(m, 'downsample'):
        bn_params, bn_state = p2j_batchnorm(m.downsample[1], f'{prefix}/shortcut_batchnorm', ndim=4)
        c_params = p2j_conv(m.downsample[0], f'{prefix}/shortcut_conv')
        params.update(**bn_params, **c_params)
        state.update(**bn_state)
    return params, state
    
def j2p_conv(prefix, params):
    return dict(weight=params[prefix]['w'].transpose((3, 2, 0, 1)), bias=params[prefix].get('b'))

def p2j_conv(m, prefix):
    assert m.bias is None
    return {prefix: dict(w=p2j_tensor(m.weight.permute(2, 3, 1, 0)))}

def add_prefix(prefix, params):
    return {f'{prefix}.{k}': v for k, v in params.items()}

def j2p_sd(sd):
    return {k: torch.from_numpy(np.asarray(v).copy()) for k, v in sd.items() if v is not None}

def j2p_mlp(prefix, params, state):
    return dict(
        **add_prefix('0', j2p_linear(f'{prefix}/linear', params)),
        **add_prefix('1', j2p_batchnorm(f'{prefix}/batch_norm', params, state)),
        **add_prefix('3', j2p_linear(f'{prefix}/linear_1', params)))

def j2p_linear(prefix, params):
    return dict(weight=params[prefix]['w'].T, bias=params[prefix].get('b'))

def MLP(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=True),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size, bias=False))

