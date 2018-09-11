import importlib
import numpy as np
import os
import torch
import copy
from bluetorch.experiment.base_experiment import BaseExperiment, BaseAnalysis
import argparse
# from defaults import fill_defaults


def default_text(args):
    # optimization parameters
    # model hyperparameters
    args.optim = 'sgd'

    # optimization parameters
    args.conv_nstep = 20
    args.momentum = 0

    args.nsamples =1
    args.iw_nsamples=500

    # select mode
    args.eval=False
    args.load_path = ''

    # annealing paramters
    args.warm_up = 10
    args.kl_start = 1.0

    # inference parameters
    args.burn=0

    # others
    args.seed=783435
    args.save_path = ''

    # these are for slurm purpose to save model
    args.jobid=0
    args.taskid=0

    args.cuda = torch.cuda.is_available()

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params

    args = argparse.Namespace(**vars(args), **params)


    return args



def fill_ais_text(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.dataset = "yahoo"
    args.mode = 'test'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = '_ais'
    #########
    args = default_text(args)
    args.eval = True
    if sub_exp == '1':
        args.description = 'Excel ais table score'
        args.exp_name = 'replVanilla'
        args.burn = 0
        args.kl_start = 1
        args.conv_nstep = 0
        args.warm_up = 10
        args.seed = 655
    elif sub_exp == '2':
        args.description = 'Excel ais table score'
        args.exp_name = 'replVanilla'
        args.burn = 0
        args.kl_start = 0.1
        args.conv_nstep = 0
        args.warm_up = 10
        args.seed = 811
    elif sub_exp == '3':
        args.description = 'Excel ais table score'
        args.exp_name = 'our'
        args.burn = 5
        args.conv_nstep = 60
        args.kl_start = [.1, 1]
        args.seed = [655, 811]

    return BaseExperiment(args)


def debug_text(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'debug'
    args.dataset = "yahoo"
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########

    args = default_text(args)
    args.eval = False
    args.burn = 1
    args.description = 'Finding what to run AIS on'

    return BaseExperiment(args)
