import os, os.path as osp
import numpy as np

import torch
import torch.nn as nn

from pix2latent.model import BigGAN

from pix2latent import VariableManager, save_variables
from pix2latent.optimizer import HybridNevergradOptimizer
from pix2latent.transform import TransformBasinCMAOptimizer, SpatialTransform
from pix2latent.utils import image, video

import pix2latent.loss_functions as LF
import pix2latent.utils.function_hooks as hook
import pix2latent.distribution as dist

on_gpu = torch.cuda.is_available()
device = 'cuda:0' if on_gpu else 'cpu'
fp = './images/dog-example-153.jpg'
mask_fp = './images/dog-example-153-mask.jpg'
class_lbl = 153
method = 'hybrid'
ng_method = 'CMA'
lr = 0.05
latent_noise = 0.05
truncate = 2.0
make_video = on_gpu
max_minibatch = 9
num_samples = 9

meta_steps_t = 50 if on_gpu else 2
grad_steps_t = 10 if on_gpu else 2
meta_steps_z = 30 if on_gpu else 1
grad_steps_z = 50 if on_gpu else 1
last_grad_steps_z = 300 if on_gpu else 1
adam_steps_z = 500 if on_gpu else 1


### ---- initialize necessary --- ###

# (1) pretrained generative model
model = BigGAN().to(device).eval()

# (2) variable creator
var_manager = VariableManager()

# (3) default l1 + lpips loss function
loss_fn = LF.ProjectionLoss()


target = image.read(fp, as_transformed_tensor=True, im_size=256)
weight = image.read(mask_fp, as_transformed_tensor=True, im_size=256)
weight = ((weight + 1.) / 2.).clamp_(0.3, 1.0)

fn = fp.split('/')[-1].split('.')[0]
save_dir = f'./results/biggan_256/{method}_{fn}_w_transform'


var_manager = VariableManager()


# (4) define input output variable structure. the variable name must match
# the argument name of the model and loss function call

var_manager.register(
            variable_name='z',
            shape=(128,),
            distribution=dist.TruncatedNormalModulo(sigma=1.0,trunc=truncate),
            var_type='input',
            learning_rate=lr,
            hook_fn=hook.Clamp(truncate),
            )

var_manager.register(
            variable_name='c',
            shape=(128,),
            default=model.get_class_embedding(class_lbl)[0],
            var_type='input',
            learning_rate=0.01,
            )

var_manager.register(
            variable_name='target',
            shape=(3, 256, 256),
            requires_grad=False,
            default=target,
            var_type='output'
            )

var_manager.register(
            variable_name='weight',
            shape=(3, 256, 256),
            requires_grad=False,
            default=weight,
            var_type='output'
            )



### ---- optimize (transformation) ---- ####

target_transform_fn = SpatialTransform(pre_align=weight)
weight_transform_fn = SpatialTransform(pre_align=weight)

tranform_params = target_transform_fn.get_default_param(as_tensor=True)

var_manager.register(
            variable_name='t',
            shape=tuple(tranform_params.size()),
            requires_grad=False,
            var_type='transform',
            grad_free=True,
            )

t_opt = TransformBasinCMAOptimizer(
            model, var_manager, loss_fn, max_batch_size=8, log=make_video)

# this tells the optimizer to apply transformation `target_transform_fn`
#  with parameter `t` on the variable `target`
t_opt.register_transform(target_transform_fn, 't', 'target')
t_opt.register_transform(weight_transform_fn, 't', 'weight')

# (highly recommended) speeds up optimization by propating information
t_opt.set_variable_propagation('z')


t_vars, (t_out, t_target, t_candidate), t_loss = \
                t_opt.optimize(meta_steps=meta_steps_t, grad_steps=grad_steps_t)


os.makedirs(save_dir, exist_ok=True)

if make_video:
    video.make_video(osp.join(save_dir, 'transform_out.mp4'), t_out)
    video.make_video(osp.join(save_dir, 'transform_target.mp4'), t_target)

image.save(osp.join(save_dir, 'transform_out.jpg'), t_out[-1])
image.save(osp.join(save_dir, 'transform_target.jpg'), t_target[-1])
image.save(osp.join(save_dir, 'transform_candidate.jpg'), t_candidate)

np.save(osp.join(save_dir, 'transform_tracked.npy'),
        {'t': t_opt.transform_tracked})

t = t_opt.get_candidate()

var_manager.edit_variable('t', {'default': t, 'grad_free': False})
var_manager.edit_variable('z', {'learning_rate': lr})


del t_opt, t_vars, t_out, t_target, t_candidate, t_loss
model.zero_grad()
if on_gpu:
    torch.cuda.empty_cache()



### ---- optimize (latent) ---- ###
from pix2latent.optimizer import GradientOptimizer
var_manager.edit_variable('z', {'grad_free': False})
opt = GradientOptimizer(
        model, var_manager, loss_fn,
        max_batch_size=max_minibatch,
        log=make_video
        )
opt.register_transform(target_transform_fn, 't', 'target')
opt.register_transform(weight_transform_fn, 't', 'weight')
vars, out, loss = opt.optimize(num_samples=num_samples, grad_steps=adam_steps_z)
# opt = HybridNevergradOptimizer(
#                 ng_method, model, var_manager, loss_fn,
#                 max_batch_size=max_minibatch,
#                 log=make_video
#                 )
#
# opt.register_transform(target_transform_fn, 't', 'target')
# opt.register_transform(weight_transform_fn, 't', 'weight')
# vars, out, loss = opt.optimize(
#                         num_samples=num_samples, meta_steps=meta_steps_z,
#                         grad_steps=grad_steps_z, last_grad_steps=last_grad_steps_z,
#                         )


### ---- save results ---- #

vars.loss = loss
save_variables(osp.join(save_dir, 'vars.npy'), vars)

if make_video:
    video.make_video(osp.join(save_dir, 'out.mp4'), out)

image.save(osp.join(save_dir, 'target.jpg'), target)
image.save(osp.join(save_dir, 'mask.jpg'), image.binarize(weight))
image.save(osp.join(save_dir, 'out.jpg'), out[-1])
np.save(osp.join(save_dir, 'tracked.npy'), opt.tracked)
