import importlib
import numpy as np
import os
import torch
import copy
from bluetorch.experiment.base_experiment import BaseExperiment, BaseAnalysis
import argparse
# from defaults import fill_defaults


def default_(args):
    # parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    args.dataset = 'omniglot'

    # optimization parameters
    args.conv_nstep =20

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args



def baseline(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'baselinetrial'
    args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########

    args = default_(args)
    args.kl_start = 1.0
    args.burn_from = 0
    ###########
    return BaseExperiment(args)

def searchbest(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'search'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########

    args = default_(args)

    ###########
    if sub_exp == 'warmup':
        args.warm_up = 10
        args.kl_start = .1
    elif sub_exp == 'steps':
        args.burn = 1
        args.conv_nstep = [5, 10, 20, 30, 100]
    elif sub_exp == 'warmsteps':
        args.burn = 1
        args.conv_nstep = [10, 20]
        args.warm_up = 10
        args.kl_start = .1
    elif sub_exp == 'burnin':
        args.description = 'Apply burnin after burn_from epochs whenever the val kl dips'
        args.eval = False
        args.burn_from = 10
        args.warm_up = 10
        args.kl_start = [.1, 1]
        args.burn = [1,5]
        args.conv_nstep = [30,50]
    elif sub_exp == 'vanilla':
        args.description = 'Vanilla no burn no annealing'
        args.burn = 0
        args.burn_from = 0
        args.kl_start = 1
        args.conv_nstep = 0
        args.warm_up = 10
    elif sub_exp == 'allburn':
        args.description = 'Apply burnin for the entire time'
        args.burn_from = 0
        args.warm_up = 10
        args.kl_start = [.1, 1]
        args.burn = 1000
        args.conv_nstep = 20

    return BaseExperiment(args)