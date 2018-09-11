import importlib
import numpy as np
import os
import torch
import copy
from bluetorch.experiment.base_experiment import BaseExperiment, BaseAnalysis
import argparse
# from defaults import fill_defaults


def default_image(args):
    args.dataset = 'omniglot'
    args.svi_steps = 10
    args.ngpu = 1
    args.gpu_ids = ''

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

    # these are for slurm purpose to save model
    args.jobid=0
    args.taskid=0

    args.cuda = torch.cuda.is_available()

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params_v2

    args = argparse.Namespace(**vars(args), **params)


    return args


def debug_image(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'debug'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########

    args = default_image(args)
    args.eval = False
    args.description = 'Debugging the SAVAE on omniglot'
    args.svi_steps = 10
    args.ngpu = 1
    # args.gpu_ids = '0,1'
    args.gpu_ids = ''

    return BaseExperiment(args)



def default_text(args):
    args.dataset = 'yahoo'
    args.svi_steps = 10
    args.ngpu = 1
    args.gpu_ids = ''

    args.nsamples =1
    args.iw_nsamples=500

    # select mode
    args.eval=False
    args.load_path = ''
    args.save_path = ''

    # annealing paramters
    args.warm_up = 10
    args.kl_start = 1.0

    # inference parameters
    args.burn=0

    # others
    args.seed=783435

    # these are for slurm purpose to save model
    args.jobid=0
    args.taskid=0

    args.cuda = torch.cuda.is_available()

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params

    args = argparse.Namespace(**vars(args), **params)


    return args


def debug_text(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'debug'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########

    args = default_text(args)
    args.eval = False
    args.description = 'Debugging the SAVAE on omniglot'
    args.svi_steps = 10
    args.epochs = 1
    args.ngpu = 1
    # args.gpu_ids = '0,1'
    args.gpu_ids = ''

    return BaseExperiment(args)

