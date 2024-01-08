"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import time
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import yaml
import warnings

# Jax
import jax
import jax.numpy as jnp

# Flax
from flax.training import train_state, checkpoints, early_stopping
import flax

# Project Code
from ds_utils import MotzkinPipeline, get_jnp_batch_ds_iter
from mps import DenseMPS, FactoredMPS, factorize_core, get_sample_generator
from neural_models import SLP
from optimizer import get_optim_sched
from train_eval_utils import TrainingEpoch, AllValidMotzkinProbMass, ClassifierEpoch
from log_fns import warning_format, SummaryWriter, dict_to_csv


def factorize_dense_model(model_cfg, init_array, rng_key):
    print("Getting params by factorizing dense model.")
    ## Build and init Dense model parameters
    dense_init_params = {}
    for k, v in model_cfg['init_params'].items():
        if k in ['core_height', 'v_bond_dim', 'skip_connection']:
            continue
        else:
            dense_init_params[k] = v
    h_bond_dim = model_cfg['init_params']['h_bond_dim']
    height = model_cfg['init_params']['core_height']
    dense_init_params['h_bond_dim'] = h_bond_dim ** height
    donor = DenseMPS(**dense_init_params)
    donor_params = donor.init(rng_key, init_array)['params']

    ## Factorize Dense parameters, and use for Factored model
    seq_len = model_cfg['init_params']['num_cores']
    factored_params = {}
    for name, params in donor_params.items():
        if name in ['core_0', f'core_{seq_len-1}']:
            num_h_legs = 1
        else:
            num_h_legs = 2

        core_expansion = factorize_core(
            name,
            params['embedding'],
            h_bond_dim,
            model_cfg['init_params']['v_bond_dim'],
            height,
            num_h_legs
        )
        factored_params = dict(factored_params, **core_expansion)
    return flax.core.frozen_dict.FrozenDict(factored_params)

def build_model(rng_key, model_cfg, optim_cfg, sched_cfg, init_shape, get_sample_gen=False):
    input_dtype = jnp.int32
    init_array = jnp.ones(init_shape, dtype=input_dtype)

    if model_cfg['name'] == "dense_mps":
        model = DenseMPS(**model_cfg['init_params'])
        params = model.init(rng_key, init_array)['params']
    elif model_cfg['name'] == "factored_mps":
        model = FactoredMPS(**model_cfg['init_params'])
        if model_cfg['init_params']['skip_connection']:
            params = model.init(rng_key, init_array)['params']
        else:
            params = factorize_dense_model(model_cfg, init_array, rng_key)

    elif model_cfg['name'] == "slp":
        model = SLP(**model_cfg['init_params'])
        params = model.init(rng_key, init_array)['params']
    else:
        raise ValueError("Choose model: dense_mps, factored_mps, slp")

    LNS_fn = model.LNS

    # Load Optimizer
    optimizer, scheduler = get_optim_sched(optim_cfg, sched_cfg)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    if get_sample_gen:
        sample_generator = get_sample_generator(model, bond_dim)
        return state, LNS_fn, scheduler, sample_generator
    else:
        return state, LNS_fn, scheduler, None


def batch_shape_iter(num_dev, start_per_dev_bs, end_per_dev_bs, seq_len, growth_len):
    for epoch in range(growth_len):
        per_dev_bs = ((growth_len-epoch)*start_per_dev_bs + epoch*end_per_dev_bs) // growth_len
        yield (num_dev, per_dev_bs, seq_len)

    while True:
        yield (num_dev, end_per_dev_bs, seq_len)


@hydra.main(version_base='1.2', config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    src_dir = hydra.utils.get_original_cwd()
    src_dir = os.path.normpath(src_dir)
    src_dir_list = src_dir.split(os.sep)
    root_dir = f"{os.sep}".join(src_dir_list[:-1]) ## Root of code dir, parent of src, outputs, data, &c.
    working_dir = os.getcwd()


    """ Set params that depend on current config
    regardless of if continuing checkpoint """
    if cfg.name[:3] == "dev":
        num_epochs = 1
        checkpoint_dir = None
    else:
        num_epochs = cfg.num_epochs
        checkpoint_dir = working_dir + "/checkpoints/"


    if cfg.chkpt.load:
        run_dir_name = input("Please input path to desired run dir (after outputs/): ")
        chkpt_num = input("Which checkpoint would you like to run?")
        old_working_dir = root_dir + "/outputs/" + run_dir_name
        checkpoint_load_dir = old_working_dir + f"/checkpoints/checkpoint_{chkpt_num}"
        start_epoch = chkpt_num + 1

        ## Switch to config from checkpoint
        chkpt_cfg_path = old_working_dir + "/hydra/config_cont.yaml"
        try:
            cfg = OmegaConf.load(chkpt_cfg_path)
            # Reload num_epochs from loaded config
            num_epochs = cfg.num_epochs
        except:
            warnings.formatwarning = warning_format
            warnings.warn(f"No new config file at {chkpt_cfg_path}. Using original config at {old_working_dir+'/hydra/config.yaml'}.")
    else:
        checkpoint_load_dir = None
        start_epoch = 0


    ## Configure params
    num_cpu = os.cpu_count()
    num_dev = jax.local_device_count()
    print(f"CPUs: {num_cpu}    Jax Dev: {num_dev} ({jax.devices()})")
    key = jax.random.PRNGKey(cfg.rng_seed)
    np.random.seed(cfg.rng_seed)
    seq_len = cfg.hparams.seq_len
    #per_dev_bs = cfg.hparams.per_dev_bs
    #train_batch_shape = (num_dev, per_dev_bs, seq_len)
    val_batch_shape = (num_dev, cfg.val_task.per_dev_bs, seq_len)
    test_batch_shape = (num_dev, cfg.test_task.batch_size, seq_len)
    train_bs_iter = batch_shape_iter(num_dev, cfg.hparams.start_per_dev_bs, cfg.hparams.end_per_dev_bs, seq_len, cfg.hparams.bs_growth_len)

    """
    Load Pipeline """
    print(f"Building pipeline for sequence length {seq_len}.")
    t0 = time.time()
    key, pipe_key = jax.random.split(key)
    data_path = root_dir + "/data/"
    valid_ds_name = f"{seq_len}_{cfg.dataset.csv_id}.csv"
    print("  Reading training data from\n    ", data_path + valid_ds_name)
    invalid_ds_name = f"{seq_len}_{cfg.test_task.neg_ds_csv_id}.csv"
    pipeline = MotzkinPipeline(data_path, valid_ds_name, invalid_ds_name, seq_len, cfg.dataset.train_size, cfg.dataset.train_pos_frac, cfg.test_task.ds_size, cfg.test_task.pos_frac)
    train_ds, val_ds, test_ds = pipeline.build(pipe_key)
    t = time.time() - t0
    print(f"    Pipeline completed ({t:.4f}s)")

    """
    Build Model: State, LNS function, Sampler, LR scheduler """
    mps_cfg = {
        'name': cfg.model.name,
        'init_params': {k: v for k, v in cfg.model.init_params.items()}
    }
    mps_cfg['init_params']['embed_dtype'] = jnp.float32
    key, rng_key = jax.random.split(key)
    init_batch_shape = (1, seq_len)
    init_state, LNS_fn, scheduler, sample_generator = build_model(rng_key, mps_cfg, cfg.optim, cfg.sched, init_batch_shape)
    state = flax.jax_utils.replicate(init_state)
    if checkpoint_load_dir is not None:
        print(f"Loading checkpoint {chkpt_num} from\n\t{old_working_dir}.")
        try:
            state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_load_dir, target=state)
        except:
            raise ValueError(f"Could not load checkpoint at {checkpoint_load_dir}.")
    model_fn_dict = {"LNS": LNS_fn, "sched": scheduler}

    """
    Logger """
    if cfg.tb_logging:
        tb_logger_path = working_dir
        tb_logger = SummaryWriter(log_dir=tb_logger_path)
        print(f"Logging Tensorboard data to\n\t{working_dir}")
    else:
        tb_logger = None

    """
    Setup epoch functions """
    metric_keys = ['loss', 'LNS', 'perplexity']
    epoch_fn_args = (model_fn_dict, tb_logger, metric_keys)
    
    # Train
    train_epoch = TrainingEpoch(seq_len, cfg.hparams.alpha, *epoch_fn_args, cfg.log_steps, cfg.log_lr)

    # Validation
    if cfg.val_task.name == "valid_motzkin_prob_mass":
        val_epoch = AllValidMotzkinProbMass("val", val_batch_shape, len(train_ds), len(val_ds), cfg.hparams.alpha, *epoch_fn_args)
    elif cfg.val_task.name == "roc_auc":
        val_epoch = get_classifier_eval(val_ds, val_batch_shape, *epoch_fn_args)
    else:
        raise ValueError(f'Validation loop "{cfg.val_task.name}" task not implemented!')

    # Test
    if cfg.test_task.name == 'roc_auc':
        test_epoch = ClassifierEpoch(test_batch_shape, LNS_fn, tb_logger)
    elif cfg.test_task.name == "sampler":
        print("Sampler task code needs to be updated/refactored!")
        raise NotImplementedError
        '''
        # Sampler Evaluation
        key, sampler_key = jax.random.split(key)
        eval_sampler = get_sampler_eval(sampler_key, chain_info_tup,
                                        sample_generator, logger)
        percent_valid, freq_arr = eval_sampler(state, num_epochs, num_histogram_samples)
        freq_dict = {"sample_freqs": freq_arr.tolist()}
        dict_to_csv(freq_dict, "sampler_histogram")
        log_histogram(freq_arr, num_epochs, chain_info_tup, logger)
        '''
    else:
        print(f"The test task, {cfg.test_task.name}, has not been implemented!")
        raise NotImplementedError

    metrics_logging = {k: [] for k in metric_keys}
    metrics_logging['epoch'] = []
    metrics_logging['trainset_prob_mass'] = []
    metrics_logging['valid_prob_mass'] = []

    def update_metrics_dict(val_epoch_metrics_np, valid_chain_probs, epoch):
        metrics_logging['epoch'].append(epoch)
        metrics_logging['trainset_prob_mass'].append(valid_chain_probs[0])
        metrics_logging['valid_prob_mass'].append(valid_chain_probs[1])
        for k, v in val_epoch_metrics_np.items():
            metrics_logging[k].append(v)
    

    """
    Learning Loop """
    valid_chain_probs = (0.0, 0.0) ## Needed with neural model, since it doesn't output any valid_chain_probs
    early_stop = early_stopping.EarlyStopping(**cfg.early_stopping)
    stopped_early = False
    '''
    ## Run pre-train validation
    if cfg.model.name[-3:] == "mps":
        val_epoch_metrics_np, valid_chain_probs = val_epoch.run(state, -1, val_ds)
        update_metrics_dict(val_epoch_metrics_np, valid_chain_probs, -1)
    '''

    ## Run training and validation loops
    for epoch in tqdm(range(start_epoch, start_epoch+num_epochs), desc="Epoch"):
        ## Set up this epoch's iterator for the training data
        train_batch_shape = next(train_bs_iter)
        tb_logger.add_scalar("per_dev_bs", train_batch_shape[1], epoch)
        train_iter = get_jnp_batch_ds_iter(train_ds, train_batch_shape, shuffle_seed=epoch)
        ## Run training epoch
        state = train_epoch.run(state, epoch, train_iter)

        if checkpoint_dir is not None and epoch % cfg.chkpt.freq == 0:
            ## Save checkpoint
            checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=epoch, keep=5)

        if cfg.model.name[-3:] == "mps" and epoch % cfg.val_task.freq == 0:
            ## Run validation epoch
            val_epoch_metrics_np, valid_chain_probs = val_epoch.run(state, epoch, val_ds)
            update_metrics_dict(val_epoch_metrics_np, valid_chain_probs, epoch)
            if valid_chain_probs[1] > 1.0:
                print(f"Reached nonsensical total valid chain probability greater than one: {valid_chain_probs[1]}. Terminating learning.")
                stopped_early = True
                break
            elif valid_chain_probs[1] >= cfg.val_task.early_stop_threshold:
                ## Stop training if mps model has total valid prob greater than threshold
                print(f"Reached {cfg.val_task.early_stop_threshold} valid chains probability mass, stopping epoch {epoch}.")
                num_epochs = epoch + 1
                stopped_early = True
                break

        _, early_stop = early_stop.update(train_epoch.metrics.mov_avg['loss'])
        if early_stop.should_stop:
            print(f"Reached early stopping criteria on train loss, stopping epoch {epoch}.")
            stopped_early = True
            break

    ## Save checkpoint & do final validation epoch
    if checkpoint_dir is not None and not stopped_early:
        checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=num_epochs, keep=5)

        val_epoch_metrics_np, valid_chain_probs = val_epoch.run(state, num_epochs, val_ds)
        update_metrics_dict(val_epoch_metrics_np, valid_chain_probs, epoch)

    ## Save validation epoch metrics to csv file
    dict_to_csv(metrics_logging, "val_epoch_metrics")

    ## Test Task
    roc_auc, prob_array, label_array = test_epoch.run(state, start_epoch+num_epochs, test_ds)
    test_data = {
        'probs': prob_array.tolist(),
        'labels': label_array.tolist(),
        'roc_auc': [roc_auc]
    }
    dict_to_csv(test_data, "roc_auc_data")

    ## Log hyperparameters
    if cfg.tb_logging:
        hp_metric_dict = {
                #"train_perplexity": perplexity,
                "train_prob_mass": valid_chain_probs[0],
                "valid_prob_mass": valid_chain_probs[1],
                "roc_auc": roc_auc
            }
        tb_logger.add_config(cfg, hp_metric_dict)


if __name__ == "__main__":
    t0 = time.time()
    main()
    t = time.time() - t0
    print(f"Total runtime: {t:.4f} s")
