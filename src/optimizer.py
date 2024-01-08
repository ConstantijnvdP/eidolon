"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import optax

def warmup_constant_schedule(init_value=0.0001, end_value=0.01, warmup_steps=1000):
    schedules = [
        optax.linear_schedule(
            init_value=init_value,
            end_value=end_value,
            transition_steps=warmup_steps),
        optax.constant_schedule(end_value)
    ]
    return optax.join_schedules(schedules, [warmup_steps])


def get_optim_sched(optim_cfg, sched_cfg):
    if sched_cfg['name'] == 'warmup_constant':
        scheduler = warmup_constant_schedule(**sched_cfg['params'])
    else:
        raise NotImplementedError("Only 'warmup_constant' scheduler is implemented!")

    if optim_cfg['name'] == 'sgd':
        optim = optax.sgd(learning_rate=scheduler, **optim_cfg['params'])
    elif optim_cfg['name'] == 'adam':
        optim = optax.adam(learning_rate=scheduler, **optim_cfg['params'])
    else:
        raise NotImplementedError(f"Selected optimizer, {optim_cfg['name']} is not implemented!")

    return optim, scheduler
