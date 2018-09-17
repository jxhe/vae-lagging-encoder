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
    args.extra_name = ''
    args.nsamples =1
    args.iw_nsamples=500

    # select mode
    args.eval=False
    args.load_path = ''

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


def beta_base_omni(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()
    args.params = argparse.Namespace()

    #########
    args.model = 'vae'
    args.mode = 'test'
    args.question = ''
    # args.extra_name = '_ais'
    #########
    args = default_(args)

    if sub_exp == '1':
        # args.eval=True
        args.description = 'Beta vae'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = [.2,.4,.6,.8]
    return BaseExperiment(args)

def beta_fill_ais_omni(sub_exp=None):
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
    args = default_(args)
    args.eval = True
    if sub_exp == '1':
        args.description = 'Beta vae'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = [.2,.4,.6,.8]
    if sub_exp == '2':
        args.description = 'Beta vae'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = .4
        args.load_path = 'models/omniglot/beta/betaVAE_omniglot_burn0_ns1_Beta0.4_seed_783435.pt'
        args.seed = [1738, 2020]
    if sub_exp == '3':
        args.description = 'Beta vae'
        args.exp_name = 'beta'
        args.burn = 0
        args.beta = .8
        args.seed = [1738, 2020]
        args.load_path = 'models/omniglot/beta/betaVAE_omniglot_burn0_ns1_Beta0.8_seed_783435.pt'

    return BaseExperiment(args)
