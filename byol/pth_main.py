from pathlib import Path
import os
import math
import argparse
import unittest
import pickle
import time
import json

import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.20'
from acme.jax import utils as acme_utils
import jax
import haiku as hk
from jax import random
import jax.numpy as jnp
import optax
import dill

import numpy as np
from PIL import Image

from byol.utils import augmentations
from byol.utils import networks
from byol.utils import dataset
from byol.utils import optimizers
from byol.utils import dataset
from byol.utils import schedules
from byol.utils import checkpointing
from byol.configs import byol as byol_config
import byol.byol_experiment
import byol.jzb_resnet

@torch.no_grad()
def update_target(model, tau):
    for tp, op in zip(model.target.parameters(), model.online.parameters()):
        tp.mul_(tau).add_(op, alpha=1 - tau)

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2

class ToHWCTensor:
    def __call__(self, pic):
        assert isinstance(pic, Image.Image)
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def main(args):
    config = byol_config.get_config(args.pretrain_epochs, args.batch_size)
    torch.backends.cudnn.benchmark = True

    tr = datasets.ImageFolder(args.data_dir / 'train',
        TwoCropsTransform(transforms.Compose([
            transforms.RandomResizedCrop(128, interpolation=Image.BICUBIC),
            ToHWCTensor(),
        ])))
    tr_loader = torch.utils.data.DataLoader(tr, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    def make_inputs(inputs):
        (view1, view2), labels = inputs
        return dict(view1=view1.numpy(), view2=view2.numpy(), labels=labels.numpy())

    model = ByolModel().cuda()
    optimizer = LARS(model.parameters(), learning_rate=None)
    rng = random.PRNGKey(0)

    if args.convert_to_jax:
        model.load_state_dict(torch.load(args.convert_to_jax, map_location='cpu'))
        byol_state = p2j_byol_model(model)
        byol_state = jax.tree_map(lambda x: jax.device_get(x), byol_state)
        checkpoint_data = dict(experiment_state=byol_state, step=36988, rng=rng)
        with open(args.tmp_dir / 'pretrain.pkl', 'wb') as checkpoint_file:
            dill.dump(checkpoint_data, checkpoint_file, protocol=2)
        exit()
    
    if False:
        print('initialize BYOL model from JAX')
        del config['network_config']['bn_config']['cross_replica_axis']
        experiment = byol.byol_experiment.ByolExperiment(**config)
        state = experiment._make_initial_state(rng, next(tr))
        sd = j2p_sd(j2p_byol_module(state))
        torch.save(sd, args.tmp_dir / 'byol_init.pth')
    model.load_state_dict(torch.load(args.tmp_dir / 'byol_init.pth', map_location='cpu'))

    start_time = last_logging = time.time()

    step = 0
    for epoch in range(args.pretrain_epochs):
        for inputs in tr_loader:
            lr = learning_schedule(global_step=step, batch_size=args.batch_size, total_steps=config['max_steps'], **config['lr_schedule_config'])
            for g in optimizer.param_groups:
                g['learning_rate'] = lr

            step_rng, rng = jax.random.split(rng)
            loss, logs = model.forward(make_inputs(inputs), step_rng)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tau = target_ema(global_step=step, base_ema=config['base_target_ema'], max_steps=config['max_steps'])
            update_target(model, tau)

            current_time = time.time()
            if current_time - last_logging > args.log_tensors_interval:
                state = dict(
                    step=step, 
                    classif_loss=logs['classif_loss'].item(),
                    learning_rate=lr, 
                    loss=loss.item(), 
                    repr_loss=logs['repr_loss'].item(),
                    tau=tau, 
                    top1_accuracy=logs['top1_accuracy'].item(), 
                    top5_accuracy=logs['top5_accuracy'].item(), 
                    time=int(current_time - start_time),
                )
                print(json.dumps(state))
                last_logging = current_time
            step += 1
    torch.save(model.state_dict(), args.tmp_dir / f'byol_model_{int(time.time())}.pth')
    
class TestBYOL(unittest.TestCase):
    FLAGS = {'random_seed': 0, 'num_classes': 10, 'batch_size': 256, 'max_steps': 36988, 'enable_double_transpose': True, 'base_target_ema': 0.996, 'network_config': {'projector_hidden_size': 4096, 'projector_output_size': 256, 'predictor_hidden_size': 4096, 'encoder_class': 'ResNet18', 'encoder_config': {'resnet_v2': False, 'width_multiplier': 1}, 'bn_config': {'decay_rate': 0.9, 'eps': 1e-05, 'create_scale': True, 'create_offset': True}}, 'optimizer_config': {'weight_decay': 1e-06, 'eta': 0.001, 'momentum': 0.9}, 'lr_schedule_config': {'base_learning_rate': 2.0, 'warmup_steps': 369}, 'evaluation_config': {'subset': 'test', 'batch_size': 25}, 'checkpointing_config': {'use_checkpointing': True, 'checkpoint_dir': '/scratch/jzb/byol_checkpoints', 'save_checkpoint_interval': 300, 'filename': 'pretrain.pkl'}}

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

        params1, state1 = p2j_resnet(m, 'res_net18/~', [2, 2, 2, 2])
        jout1, _ = forward.apply(params1, state1, x, True)
        assert allclose(jout, j2p_tensor(jout1))

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

        params1, state1 = p2j_blockgroup(m, 'res_net18/~/block_group_3/~', num_blocks=2)
        jout1, _ = forward.apply(params1, state1, x, True)
        assert allclose(jout, j2p_tensor(jout1))


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

        exp = byol.byol_experiment.ByolExperiment(**TestBYOL.FLAGS)
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

        params1, state1 = p2j_byol_network(m)
        jout1, _ = exp.forward.apply(params1, state1, input, is_training=True)
        for k in jout.keys():
            assert allclose(jout[k], j2p_tensor(jout1[k]))
    
    def test_loss_fn(self):
        k0, k1, k2, k3 = random.split(random.PRNGKey(0), 4)
        batch_size = 4
        input = dict(
            view1=random.normal(k1, (batch_size, 128, 128, 3)),
            view2=random.normal(k2, (batch_size, 128, 128, 3)),
            labels=random.randint(k3, (batch_size,), 0, 9))

        exp = byol.byol_experiment.ByolExperiment(**TestBYOL.FLAGS)
        state = exp._make_initial_state(k0, input)
        jout = exp.loss_fn(rng=k0, inputs=input, 
            online_params=state.online_params, 
            online_state=state.online_state, 
            target_params=state.target_params, 
            target_state=state.target_state)

        m = ByolModel().cuda()
        sd = j2p_sd(j2p_byol_module(state))
        m.load_state_dict(sd)
        tout = m(input, k0)

        jout = jout[1][1]
        tout = tout[1]

        assert jout.keys() == tout.keys()
        for k in jout.keys():
            assert allclose(jout[k], tout[k])

    def grad_linear(self):
        def _forward(inputs):
            return hk.Linear(output_size=4, with_bias=True, b_init=hk.initializers.RandomNormal())(inputs)

        def _loss(params, state, k, inputs):
            jout, _ = forward.apply(params, state, inputs)
            return jnp.mean(jout)

        def torch_grad():
            loss = m(j2p_tensor(inputs)).mean()
            m.zero_grad()
            loss.backward()

        forward = hk.without_apply_rng(hk.transform_with_state(_forward))
        k = random.PRNGKey(0)
        inputs = random.normal(k, (2, 3))
        params, state = forward.init(k, inputs)
        grad_fn = jax.grad(_loss)
        grad = grad_fn(params, state, k, inputs)

        m = nn.Linear(3, 4).cuda()
        m.load_state_dict(j2p_sd(j2p_linear('linear', params)))
        torch_grad()

        return params, state, grad, m, torch_grad

    def test_grad_linear(self):
        grad, m = self.grad_linear()
        grad1 = p2j_linear(m, 'linear', grad=True)

        assert allclose(grad['linear']['w'], grad1['linear']['w'])
        assert allclose(grad['linear']['b'], grad1['linear']['b'])

    def helper_test_gradient_transform(self, jt, pt):
        params, state, grads, m, torch_grad = self.grad_linear()
        jstate = jt.init(params)
        pt.init(m.parameters())
        for i in range(3):
            print(i)
            jgrads, jstate = jt.update(grads, jstate, params)
            pt.update()
            tgrads = p2j_linear(m, 'linear', grad=True)
            torch_grad()
            assert allclose(jgrads, tgrads)

    def test_LARS(self):
        params, state, grads, m, _ = self.grad_linear()
        kwargs = dict(learning_rate=0.1, weight_decay=1e-2, momentum=0.9, eta=0.001)
        pt = LARS(m.parameters(), weight_decay_filter=exclude_bias_and_norm, 
                  lars_adaptation_filter=exclude_bias_and_norm, **kwargs)
        jt = optimizers.lars(weight_decay_filter=optimizers.exclude_bias_and_norm,
                             lars_adaptation_filter=optimizers.exclude_bias_and_norm, **kwargs)
        jstate = jt.init(params)
        for i in range(5):
            print(i)
            pt.step()
            updates, jstate = jt.update(grads, jstate, params)
            params = optax.apply_updates(params, updates)
            assert allclose(params, p2j_linear(m, 'linear'))

    def test_cosine_decay(self):
        self.assertAlmostEqual(schedules._cosine_decay(jnp.array([3]), 10, 2)[0], cosine_decay(3, 10, 2))
        self.assertAlmostEqual(schedules._cosine_decay(jnp.array([30]), 10, 2)[0], cosine_decay(30, 10, 2))
        self.assertAlmostEqual(schedules._cosine_decay(jnp.array([0]), 10, 2)[0], cosine_decay(0, 10, 2))
        self.assertAlmostEqual(schedules._cosine_decay(jnp.array([10]), 100, 4)[0], cosine_decay(10, 100, 4))

    def test_learning_schedule(self):
        config = dict(batch_size=4096, base_learning_rate=3, total_steps=20, warmup_steps=5)
        for step in range(config['total_steps']):
            jout = learning_schedule(step, **config)
            tout = schedules.learning_schedule(jnp.array([step]), **config)[0]
            self.assertAlmostEqual(tout, jout, places=4)

    def test_target_ema(self):
        config = dict(base_ema=0.9, max_steps=20)
        for step in range(config['max_steps']):
            jout = target_ema(step, **config)
            tout = schedules.target_ema(jnp.array([step]), **config)[0]
            self.assertAlmostEqual(tout, jout, places=6)

def target_ema(global_step, base_ema, max_steps):
    decay = cosine_decay(global_step, max_steps, 1.)
    return 1. - (1. - base_ema) * decay

def learning_schedule(global_step, batch_size, base_learning_rate, total_steps, warmup_steps):
    scaled_lr = base_learning_rate * batch_size / 256.
    learning_rate = global_step / warmup_steps * scaled_lr if warmup_steps > 0 else scaled_lr
    if global_step < warmup_steps:
        return learning_rate
    else:
        return cosine_decay(global_step - warmup_steps, total_steps - warmup_steps, scaled_lr)

def cosine_decay(global_step, max_steps, initial_value):
    global_step = min(global_step, max_steps)
    cosine_decay_value = 0.5 * (1 + math.cos(math.pi * global_step / max_steps))
    return initial_value * cosine_decay_value

def j2p_tensor(x):
    y = torch.from_numpy(np.asarray(x).copy())
    if y.dtype == torch.int32:
        y = y.long()
    if y.ndim == 4:
        y = y.permute(0, 3, 1, 2)
    y = y.cuda()
    return y

def allclose(jx, tx, **kwargs):
    if isinstance(tx, dict):
        assert jx.keys() == tx.keys()
        return all(allclose(jx[k], tx[k]) for k in jx.keys())
    if isinstance(jx, jnp.ndarray):
        jx = j2p_tensor(jx)
    if isinstance(tx, jnp.ndarray):
        tx = j2p_tensor(tx)
    close = torch.allclose(jx, tx, **kwargs)
    if not close:
        print((jx - tx).abs().max())
    return close


class LARS(optim.Optimizer):
    def __init__(self, params, learning_rate, weight_decay=0, momentum=0.9, eta=0.001, 
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum, eta=eta, 
                        weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0., 
                                    torch.where(update_norm > 0, 
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    mu = param_state['mu'] = torch.zeros_like(p)
                else:
                    mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['learning_rate'])

def exclude_bias_and_norm(p):
    return p.ndim == 1

def j2p_batchnorm(prefix, params, state):
    # assert state[f'{prefix}/~/var_ema']['counter'] == 0
    return dict(
        weight=params[prefix]['scale'].ravel(), 
        bias=params[prefix]['offset'].ravel(), 
        running_mean=state[f'{prefix}/~/mean_ema']['hidden'].ravel(), 
        running_var=state[f'{prefix}/~/var_ema']['hidden'].ravel(),
        num_batches_tracked=0)

def p2j_batchnorm(m, prefix, batch_size=256, ndim=2):
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
        f'{prefix}/~/var_ema': ema((batch_size - 1) / batch_size * m.running_var),
    }
    return params, state

def p2j_tensor(x):
    return jnp.array(x.detach().cpu().numpy())

def p2j_linear(m, prefix, grad=False):
    params = dict(w=p2j_tensor(m.weight.grad.T if grad else m.weight.T))
    if m.bias is not None:
        params['b'] = p2j_tensor(m.bias.grad if grad else m.bias)
    return {prefix: params}

def p2j_mlp(m, prefix):
    bn_params, bn_state = p2j_batchnorm(m[1], f'{prefix}/batch_norm')
    params = {
        **p2j_linear(m[0], f'{prefix}/linear'),
        **bn_params,
        **p2j_linear(m[3], f'{prefix}/linear_1')}
    return params, bn_state

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

class ByolModel(nn.Module):
    postprocess_jit = jax.jit(augmentations.postprocess)

    def __init__(self):
        super().__init__()
        self.online = ByolNetwork()
        self.target = ByolNetwork()

    def forward(self, inputs, rng):
        inputs = ByolModel.postprocess_jit(inputs, rng)
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

def p2j_byol_model(m):
    op, os = p2j_byol_network(m.online)
    tp, ts = p2j_byol_network(m.target)
    return byol.byol_experiment._ByolExperimentState(opt_state=None,
        online_params=op, online_state=os,
        target_params=tp, target_state=ts)

def j2p_byol_network(params, state):
    return dict(
        **add_prefix('classifier', j2p_linear('classifier', params)),
        **add_prefix('predictor', j2p_mlp('predictor', params, state)),
        **add_prefix('projector', j2p_mlp('projector', params, state)),
        **add_prefix('net', j2p_resnet('res_net18/~', params, state)),
    )

def p2j_byol_network(m):
    p0 = p2j_linear(m.classifier, 'classifier')
    p1, s1 = p2j_mlp(m.predictor, 'predictor')
    p2, s2 = p2j_mlp(m.projector, 'projector')
    p3, s3 = p2j_resnet(m.net, 'res_net18/~', [2, 2, 2, 2])
    return {**p0, **p1, **p2, **p3}, {**s1, **s2, **s3}

def j2p_resnet(prefix, params, state):
    return dict(
        **add_prefix('conv1', j2p_conv(f'{prefix}/initial_conv', params)),
        **add_prefix('bn1', j2p_batchnorm(f'{prefix}/initial_batchnorm', params, state)),
        **add_prefix('layer1', j2p_blockgroup(f'{prefix}/block_group_0/~', params, state)),
        **add_prefix('layer2', j2p_blockgroup(f'{prefix}/block_group_1/~', params, state)),
        **add_prefix('layer3', j2p_blockgroup(f'{prefix}/block_group_2/~', params, state)),
        **add_prefix('layer4', j2p_blockgroup(f'{prefix}/block_group_3/~', params, state)),
    )

def p2j_resnet(m, prefix, num_blocks):
    c1_params = p2j_conv(m.conv1, f'{prefix}/initial_conv')
    bn1_params, bn1_state = p2j_batchnorm(m.bn1, f'{prefix}/initial_batchnorm', ndim=4)
    p1, s1 = p2j_blockgroup(m.layer1, f'{prefix}/block_group_0/~', num_blocks[0])
    p2, s2 = p2j_blockgroup(m.layer2, f'{prefix}/block_group_1/~', num_blocks[1])
    p3, s3 = p2j_blockgroup(m.layer3, f'{prefix}/block_group_2/~', num_blocks[2])
    p4, s4 = p2j_blockgroup(m.layer4, f'{prefix}/block_group_3/~', num_blocks[3])
    params = {**c1_params, **bn1_params, **p1, **p2, **p3, **p4}
    state = {**bn1_state, **s1, **s2, **s3, **s4}
    return params, state

def j2p_blockgroup(prefix, params, state):
    return dict(
        **add_prefix('0', j2p_block(f'{prefix}/block_0/~', params, state)),
        **add_prefix('1', j2p_block(f'{prefix}/block_1/~', params, state)),
    )

def p2j_blockgroup(m, prefix, num_blocks):
    params, state = {}, {}
    for i in range(num_blocks):
        p, s = p2j_block(m[i], f'{prefix}/block_{i}/~')
        params.update(**p)
        state.update(**s)
    return params, state

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
    if m.downsample is not None:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pretrain-epochs', type=int, default=1000)
    parser.add_argument('--log-tensors-interval', type=int, default=60)
    parser.add_argument('--convert-to-jax', type=Path)
    parser.add_argument('--tmp-dir', type=Path, default='/checkpoint/jzb/tmp')
    parser.add_argument('--data-dir', type=Path, default='/checkpoint/jzb/imagenette2-160')
    args = parser.parse_args()

    main(args)
