"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
from math import prod
import numpy as np
from functools import partial
from sklearn.metrics import roc_auc_score
import warnings
from log_fns import warning_format

import jax
import jax.numpy as jnp
from jax import pmap
from flax import jax_utils

from ds_utils import get_jnp_batch_ds_iter
from motzkin_ds import decode_samp


## Loss and Metrics
def calc_log_probs(log_outputs, log_norm_sq):
    return 2*log_outputs - log_norm_sq


def cross_ent(log_outputs, log_norm_sq, bin_label):
    log_probs = calc_log_probs(log_outputs, log_norm_sq)
    pos_loss = -jnp.mean(log_probs * bin_label[:, None])
    """ V replace below with mask? V """
    neg_probs = jnp.exp(log_probs) * (1-bin_label)[:, None]
    neg_loss = -jnp.mean(jnp.log(1 - neg_probs))
    return pos_loss + neg_loss

def get_loss(alpha, binary_labels=False):
    if binary_labels:
        def loss_fn(log_outputs, log_norm_sq, bin_label):
            Xent = cross_ent(log_outputs, log_norm_sq, bin_label)
            return Xent + alpha * log_norm_sq
    else:
        def loss_fn(log_outputs, log_norm_sq):
            logprob_loss = -jnp.mean(calc_log_probs(log_outputs, log_norm_sq))
            return logprob_loss + alpha * log_norm_sq

    return loss_fn


'''
def grad_coherence(grads):
    norm_of_mean = jnp.linalg.norm(grads.mean(axis=0))
    mean_of_norm = np.linalg.norm(grads, axis=1).mean()

    return norm_of_mean / mean_of_norm
'''


class EpochMetrics:
    def __init__(self, mov_avg_keys=None, tots_keys=None):
        self.steps = 0
        if mov_avg_keys is not None:
            self.mov_avg = {key: 0.0 for key in mov_avg_keys}
        else:
            self.mov_avg = None
        if tots_keys is not None:
            self.tots = {key: 0.0 for key in tots_keys}
        else:
            self.tots = None

    def update(self, batch_steps, mov_avg_dict=None, tots_dict=None):
        self.steps += batch_steps

        if mov_avg_dict is not None:
            for key, metric in mov_avg_dict.items():
                prev = self.mov_avg[key]
                self.mov_avg[key] = prev + batch_steps*(metric - prev)/self.steps

        if tots_dict is not None:
            for key, metric in tots_dict.items():
                self.tots[key] += metric

    def reset(self):
        self.steps = 0
        if self.mov_avg is not None:
            self.mov_avg = {key: 0.0 for key in self.mov_avg.keys()}
        if self.tots is not None:
            self.tots = {key: 0.0 for key in self.tots.keys()}


class Epoch:
    def __init__(self, task_id, model_fn_dict, logger, mov_avg_keys, tots_keys, log_steps, log_lr):
        self.task_id = task_id
        self.sched = model_fn_dict['sched']

        self.p_axis = 'num_devices' ## Axis used for pmap

        self.logger = logger
        self.metrics = EpochMetrics(mov_avg_keys=mov_avg_keys, tots_keys=tots_keys)
        self.log_steps = log_steps ## Boolean: whether to log each training step (True), or just epoch metrics (False).
        self.log_lr = log_lr ## Boolean: whether to log learning rate each step

    def step(self, state, batch, setup):
        raise NotImplementedError

    def epoch_setup(self, state):
        raise NotImplementedError

    def run(self, state, epoch_num, ds_iter):
        self.metrics.reset()
        setup = self.epoch_setup(state)
        """Train for a single epoch."""
        for batch in ds_iter():
            state, loss, LNS = self.step(state, batch, setup)

            mov_avg_dict = {
                'loss': loss,
                'LNS': LNS,
                'perplexity': jnp.exp(loss/self.seq_len)
            }
            self.metrics.update(prod(batch['input'].shape), mov_avg_dict=mov_avg_dict)

            if self.log_steps:
                assert self.logger is not None, "Need a logger to log steps!"
                self.logger.add_scalar_dict(mov_avg_dict, state.step[0], tag=f"{self.task_id}")
            if self.log_lr:
                assert self.logger is not None, "Need a logger to log learning rate!"
                current_lr = self.sched(state.step[0])
                self.logger.add_scalar("learning_rate", current_lr, state.step[0])

        if self.logger is not None:
            self.logger.add_scalar_dict(self.metrics.mov_avg, epoch_num, tag=f"{self.task_id}/epoch")

        return state


## Training Loop
def get_mps_grad_fn(alpha, LNS_fn):
    loss_fn = get_loss(alpha, binary_labels=False)

    def train_loss(params, state, batch):
        log_outputs = state.apply_fn({'params': params}, batch['input'])
        log_norm_sq = LNS_fn(params)
        loss = loss_fn(log_outputs, log_norm_sq)
        return loss, log_norm_sq

    return jax.value_and_grad(train_loss, has_aux=True)  ## Differentiates with respect to first arg of train_loss (params)


class TrainingEpoch(Epoch):
    def __init__(self, seq_len, alpha, model_fn_dict, logger, mov_avg_keys, log_steps=False, log_lr=False):
        super().__init__("train", model_fn_dict, logger, mov_avg_keys, None, log_steps, log_lr)
        self.seq_len = seq_len
        
        grad_fn = get_mps_grad_fn(alpha, model_fn_dict['LNS'])

        def p_step(state, batch):
            """Train for a single step."""
            (p_loss, p_LNS), p_grads = grad_fn(state.params, state, batch)
            sync_grads = jax.lax.pmean(p_grads, axis_name=self.p_axis)
            sync_loss = jax.lax.pmean(p_loss, axis_name=self.p_axis)
            state = state.apply_gradients(grads=sync_grads)
            #grad_coh = jax.lax.pmean(grad_coherence(p_grads), axis_name=self.p_axis)
            """
            N.B. pmean syncs over devices, so we can just take the first element,
            which will be the same on each device in the ShardedDeviceArray
            """
            return state, sync_loss, p_LNS

        self.p_step = pmap(p_step, axis_name=self.p_axis)

    def step(self, state, batch, setup):
        state, sync_loss, p_LNS = self.p_step(state, batch)
        loss = jax_utils.unreplicate(sync_loss)
        LNS = jax_utils.unreplicate(p_LNS)
        return state, loss, LNS

    def epoch_setup(self, state):
        return None


## Validation / Test Loop
class SumProbValEpoch(Epoch):
    def __init__(self, task_id, seq_len, alpha, model_fn_dict, logger, mov_avg_keys, log_steps=False, log_lr=False):
        super().__init__(task_id, model_fn_dict, logger, mov_avg_keys, ["prob_mass"], log_steps, log_lr)
        self.seq_len = seq_len

        self.LNS_fn = model_fn_dict['LNS']
        loss_fn = get_loss(alpha, binary_labels=True)

        def p_step(state, batch, LNS):
            log_outputs = state.apply_fn({'params': state.params}, batch['input'])
    
            log_probs = calc_log_probs(log_outputs, LNS)
            sync_prob_sum = jax.lax.psum(jnp.exp(log_probs).sum(), axis_name=self.p_axis)
    
            p_loss = loss_fn(log_outputs, LNS, batch['label'])
            sync_loss = jax.lax.pmean(p_loss, axis_name=self.p_axis)
    
            return sync_loss, sync_prob_sum

        self.p_step = pmap(p_step, axis_name=self.p_axis)

    def epoch_setup(self, state):
        ## Return replicated LNS value for epoch
        return jax_utils.replicate(self.LNS_fn(jax_utils.unreplicate(state.params)))

    def step(self, state, batch, LNS):
        sync_loss, sync_prob_sum = self.p_step(state, batch, LNS)
        loss = jax_utils.unreplicate(sync_loss)
        batch_steps = prod(batch['label'].shape)
        prob_mass = {'prob_mass': jax_utils.unreplicate(sync_prob_sum)}
        self.metrics.update(batch_steps, tots_dict=prob_mass)
        return state, loss, LNS


class AllValidMotzkinProbMass:
    def __init__(self, task_id, batch_shape, train_len, pos_len, alpha, model_fn_dict, logger, mov_avg_keys, log_steps=False, log_lr=False):
        self.task_id = task_id

        seq_len = batch_shape[-1]
        ## Setup Epoch for training dataset, and for remaining valid Motzkin chains
        self.train_probmass = SumProbValEpoch("train_val", seq_len, alpha, model_fn_dict, None, mov_avg_keys, log_steps=False, log_lr=False)
        self.remain_probmass = SumProbValEpoch("remain_val", seq_len, alpha, model_fn_dict, None, mov_avg_keys, log_steps=False, log_lr=False)

        self.batch_shape = batch_shape
        self.train_len = train_len
        self.pos_len = pos_len

        self.metrics = EpochMetrics(mov_avg_keys=mov_avg_keys, tots_keys=['prob_mass'])
        self.logger = logger

    def epoch_probmass(self, state, epoch, ds, epoch_class, start_idx, stop_idx):
        ds_iter = get_jnp_batch_ds_iter(ds.select(range(start_idx, stop_idx)), self.batch_shape)
        epoch_class.run(state, epoch, ds_iter)
        mov_avg_dict = epoch_class.metrics.mov_avg
        tots_dict = epoch_class.metrics.tots
        self.metrics.update(stop_idx-start_idx, mov_avg_dict=mov_avg_dict, tots_dict=tots_dict)

    def run(self, state, epoch, eval_ds):
        self.metrics.reset()
        ## Validate samples in training set first
        self.epoch_probmass(state, epoch, eval_ds, self.train_probmass, 0, self.train_len)
        trainset_probmass = self.metrics.tots['prob_mass']

        ## Validate remaining positive samples
        self.epoch_probmass(state, epoch, eval_ds, self.remain_probmass, self.train_len, self.pos_len)
        pos_probmass = self.metrics.tots['prob_mass']

        if self.logger is not None:
            self.logger.add_scalar("valid_chain_probs/trainset_prob_mass", trainset_probmass, epoch)
            self.logger.add_scalar("valid_chain_probs/valid_prob_mass", pos_probmass, epoch)
            self.logger.add_scalar_dict(self.metrics.mov_avg, epoch, tag=f"{self.task_id}/epoch")

        valid_chain_probs = (trainset_probmass, pos_probmass)

        return self.metrics.mov_avg, valid_chain_probs


class ClassifierEpoch:
    def __init__(self, batch_shape, LNS_fn, logger):
        assert batch_shape[0] == 1, "Classifier eval only compatible with one device!"
        assert len(batch_shape) == 3, f"Batch shape must have three elements (per dev batch size, seq_len)! System input {batch_shape}."
        self.batch_shape = batch_shape
        self.LNS_fn = jax.jit(LNS_fn)
        self.logger = logger

        def step(state, input_batch, LNS):
            log_outputs = state.apply_fn({'params': state.params}, input_batch)
            return calc_log_probs(log_outputs, LNS)

        self.step = jax.jit(step)

    def run(self, state, epoch, eval_ds):
        state = jax_utils.unreplicate(state)
        LNS = self.LNS_fn(state.params)

        prob_array = np.zeros(len(eval_ds))
        label_array = np.zeros(len(eval_ds))

        eval_iter = get_jnp_batch_ds_iter(eval_ds, self.batch_shape)
        for idx, batch in enumerate(eval_iter()):
            log_probs = self.step(state, batch['input'][0], LNS) 
            batch_size = batch['input'].shape[1]
            prob_array[idx*batch_size: (idx+1)*batch_size] = jnp.exp(log_probs)
            label_array[idx*batch_size: (idx+1)*batch_size] = batch['label']

        not_nan = ~np.isnan(prob_array)
        num_nan = (~not_nan).sum()
        if num_nan > 0:
            warnings.formatwarning = warning_format  
            warnings.warn(f"Detected {num_nan} NaN values in model output")

        try:
            roc_auc = roc_auc_score(label_array[not_nan], prob_array[not_nan])
        except:
            print("Issue calculating ROC AUC")
            print("Labels:", label_array)
            print("Model outputs:", prob_array)
            print("Not NaN mask:", not_nan)
            roc_auc = 0

        return roc_auc, prob_array, label_array


## Sampler evaluation
def get_sampler_eval(key, chain_info_tup, sample_generator, logger):
    """
        Set up sampler evaluation
        chain_info_tup: (ordered list of chains, length of training set, number of valid chains)
    """
    """
        generate chain_list from dataset, don't store in memory
    """
    chain_ord = {chain: i for i, chain in enumerate(chain_list)}
    train_len = chain_info_tup[0]
    num_pos = chain_info_tup[1]

    tag = "sampler_eval/"

    def eval_samples(state, epoch, num_samples):
        unrep_state = jax_utils.unreplicate(state)
        freq_arr = np.zeros(len(chain_info_tup[0]), dtype=np.float32)
        for step, sample in sample_generator(unrep_state.params, key, num_samples):
            sample_str = decode_samp(sample)
            index = chain_ord[sample_str]
            freq_arr[index] += 1

        tot_valid = freq_arr[:num_pos].sum()
        percent_valid = tot_valid / num_samples
        logger.add_scalar(tag+"percent_valid", percent_valid, epoch)

        percent_val_in_train = freq_arr[:train_len].sum() / tot_valid
        logger.add_scalar(tag+"percent_valid_in_train", percent_val_in_train, epoch)

        ent = -np.sum(freq_arr * np.log(freq_arr + 1E-8))/tot_valid + np.log(tot_valid)
        logger.add_scalar(tag+"sampler_entropy", ent, epoch)

        return percent_valid, freq_arr

    return eval_samples
