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
    args.nsamples = 1
    args.iw_nsamples = 500

    # select mode
    args.eval=False
    args.load_path = ''
    # args.save_path = ''#in config

    args.burn = 0
    # args.ngpu = 1
    # args.gpu_ids = 0

    # others
    # args.seed=783435
    # args.train_from = ''
    # these are for slurm purpose to save model
    # args.jobid=0
    # args.taskid=0

    args.cuda = torch.cuda.is_available()

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params_v2 #v2 because of lists of lists

    args = argparse.Namespace(**vars(args), **params)
    return args

def default_text(args):
    args.optim = 'sgd'

    args.momentum = 0
    args.nsamples = 1
    args.iw_nsamples = 500

    # select mode
    args.eval=False
    args.load_path = ''
    args.save_path = ''

    # annealing paramters
    args.warm_up = 10

    args.burn = 0
    # args.ngpu = 1
    # args.gpu_ids = 0

    # others
    # args.seed=783435
    args.train_from = ''

    args.cuda = torch.cuda.is_available()

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    return args


def omni_beta_seeds(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()#doesn't matter
    args.params = argparse.Namespace()#doesn't matter

    #########
    args.model = 'betavae'#doesn't matter
    args.mode = 'test'#doesn't matter
    args.exp_name = 'review_beta_seeds' #/dataset/exp_name
    args.description = 'uncertainty'
    args.question = ''
    args.extra_name = ''
    args = default_image(args)
    #########
    args.seed = [783435, 101, 202, 303, 404]
    args.beta = [0.2, 0.4, 0.6, 0.8]
    return BaseExperiment(args)


def text_beta_seeds(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()#doesn't matter
    args.params = argparse.Namespace()#doesn't matter

    #########
    args.model = 'betavae'#doesn't matter
    args.mode = 'test'#doesn't matter
    args.exp_name = 'review_beta_seeds' #/dataset/exp_name
    args.description = 'uncertainty'
    args.question = ''
    args.extra_name = ''
    #########
    if sub_exp == 'yelp':
        args.dataset = 'yelp'

    if sub_exp == 'yahoo':
        args.label = False
        args.dataset = 'yahoo'
    args = default_text(args)#XXXmake sure that all grid things happen after this line

    args.seed = [783435, 101, 202, 303, 404]
    args.beta = [0.2, 0.4, 0.6, 0.8]
    return BaseExperiment(args)