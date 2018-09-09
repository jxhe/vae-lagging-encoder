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
    args.seed = 666
    ###########
    return BaseExperiment(args)


def reproduce_vanilla(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'replVanilla'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########
    args = default_(args)

    if sub_exp == 'vanilla':
        args.description = 'Vanilla replicate junhians baseline results'
        args.eval = True
        args.burn = 0
        args.burn_from = 0
        args.kl_start = [0.1, 1]
        args.conv_nstep = 0
        args.warm_up = 10
        args.seed = [811, 332, 345, 655]
        # collapsed: 558, 309
    return BaseExperiment(args)

def debug(sub_exp=None):
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

    args = default_(args)
    args.description = 'Getting image and image_v to be the same'

    return BaseExperiment(args)


def our_method(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.exp_name = 'our'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = ''
    #########

    args = default_(args)

    args.description = 'Seeing how our method works on two seeds that correspond to V-VAE experiments'
    args.burn_from = 0
    args.burn = 5
    args.conv_nstep = 60
    args.kl_start = [.1, 1]
    args.seed = [655, 811]
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
        args.description = 'Reproducing experiment from excel sheet'
        args.burn = 5
        args.burn_from = 0
        args.conv_nstep = 60
        args.seed = [XXX, XXX]
    elif sub_exp == 'warmsteps':
        args.burn = 1
        args.conv_nstep = [10, 20]
        args.warm_up = 10
        args.kl_start = .1
    elif sub_exp == 'burnin':
        args.description = 'Apply burnin after burn_from epochs whenever the val kl dips'
        args.eval = True
        args.burn_from = 10
        args.warm_up = 10
        args.kl_start = [.1, 1]
        args.burn = [1,5]
        args.conv_nstep = [30,50]
    elif sub_exp == 'allburn':
        args.description = 'Apply burnin for the entire time'
        args.burn_from = 0
        args.warm_up = 10
        args.kl_start = [.1, 1]
        args.burn = 1000
        args.conv_nstep = 20

    return BaseExperiment(args)