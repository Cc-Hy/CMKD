from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle
from pcdet.models.backbones_3d.vfe.image_vfe import ImageVFE


def build_optimizer(model, optim_cfg):

    if optim_cfg.get('FREEZE_BACKBONE', False) is True:
        for i,j in enumerate(list(model.children())):       
            if isinstance(j,(ImageVFE)):
                for k in j.parameters():
                    k.requires_grad = False

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=optim_cfg.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):

    lr_warmup_scheduler = None

    total_steps = total_iters_each_epoch * total_epochs

    # if adam_onecycle, use the original OpenPCDet optimizer
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    # if not, use official pytorch onecycle scheduler
    elif optim_cfg.get('SCHEDULER',None) == 'OneCycleLR':
        lr_scheduler = lr_sched.OneCycleLR(
            optimizer, max_lr=optim_cfg.LR, total_steps=total_steps, pct_start=optim_cfg.PCT_START, 
            base_momentum=optim_cfg.BASE_MOM, max_momentum=optim_cfg.MAX_MOM, div_factor=optim_cfg.START_RATE, final_div_factor=optim_cfg.END_RATE,)
    # else, use DECAY_STEP_LIST
    else:
        decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * optim_cfg.LR_DECAY
            return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
