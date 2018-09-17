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
    args.momentum = 0
    args.extra_name = ''
    # optimization parameters
    args.conv_nstep = 20
    args.momentum = 0

    args.nsamples =1
    args.iw_nsamples=500

    # select mode
    args.eval=False
    args.load_path = ''


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

def beta_base_yahoo(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    #########
    args.dataset = "yahoo"
    args.label = False
    args = default_text(args)
    if sub_exp == '1':
        args.description = 'Best baselines on yahoo'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = [0.2,0.4,0.6,0.8]

    return BaseExperiment(args)

def beta_base_yelp(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    #########
    args.dataset = "yelp"
    args = default_text(args)
    if sub_exp == '1':
        args.eval = True
        args.extra_name = '_ais'
        args.description = 'Best baselines on yelp'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = [0.2,0.4,0.6,0.8]
    return BaseExperiment(args)


def beta_fill_ais_text_yelp(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = '_ais'
    #########
    args.dataset = "yelp"
    args = default_text(args)
    args.eval = True
    args.label
    foooo
    return BaseExperiment(args)

def beta_fill_ais_text_yahoo(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    # args.description = 'Vanilla VAE Baseline'
    args.question = ''
    args.extra_name = '_ais'
    #########
    args.dataset = "yahoo"
    args.label = False
    args = default_text(args)
    args.eval = True
    if sub_exp == '1':
        args.description = 'Best baselines on yahoo'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = [0.2,0.4,0.6,0.8]
    return BaseExperiment(args)
