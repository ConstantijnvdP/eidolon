"""
Copyright 2023 Constantijn van der Poel
SPDX-License-Identifier: Apache-2.0
"""
import csv
import os
import matplotlib.pyplot as plt
import tensorboardX


def warning_format(message, category, filename, lineno, file=None, line=None):
    return f'\n{category.__name__}:\n{message}\n'


class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self, log_dir=None, **kwargs):
        super().__init__(log_dir, **kwargs)

    # takes dictionary as input and combines all plots into one subgroup by a tag
    def add_scalar_dict(self, metric_dict, global_step, tag=None):
        for key, val in metric_dict.items():
            if tag is not None:
                key = os.path.join(tag, key)
            self.add_scalar(key, val, global_step)

    def add_config(self, cfg, metric_dict):
        hparam_dict = {}
        sub_dicts = ['hparams', 'optim', 'sched']
        for sub in sub_dicts:
            hparam_dict = dict(hparam_dict, **cfg[sub])
        hparam_dict = dict(hparam_dict, **cfg.model.init_params)
        hparam_dict['model_name'] = cfg.model.name
        self.add_hparams(hparam_dict, metric_dict, global_step=0)


def dict_to_csv(metric_dict, filename):
    with open(filename+'.csv', 'w') as f:
        w = csv.writer(f)
        for k, v in metric_dict.items():
            w.writerow([k] + v)


def log_histogram(freq_arr, step, chain_info_tup, logger, x_label='Chain', y_label='Count', tag="e    val_sampling_hist"):
    x = chain_info_tup[0]
    train_len = chain_info_tup[1]
    num_pos = chain_info_tup[2]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 7.5))
    ax = plt.axes()

    plt.bar(x, freq_arr, width=1.0, color="blue")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=90)

    ## Draw separators for train set, and positive data
    plt.axvline(x=train_len-0.5, color='red', linestyle='--')
    plt.axvline(x=num_pos-0.5, color='red', linestyle='--')

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    logger.add_figure(tag, fig, step)


'''
from motzkin_ds import decode_samp

def setup_sample_logger(sample_generator, logger):
    """
        Will need to change effective bond_dim for factored MPS model
    """
    def log_samples(state, key, num_samples, tag):
        for step, tokens in sample_generator(state.params, key, num_samples):
            datum = decode_samp(tokens)
            logger.add_text(tag, datum, step)

    return log_samples
'''
