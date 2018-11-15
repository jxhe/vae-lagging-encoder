import importlib
import numpy as np
import os
import torch
import copy
from bluetorch.experiment.base_experiment import BaseExperiment, BaseAnalysis
import argparse
# from defaults import fill_defaults



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

    args.burn = 0
    args.seed = 783435
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


def text_anneal(sub_exp=None):
    print(sub_exp, "sub_exp")
    args = argparse.Namespace()
    args.options = argparse.Namespace()#doesn't matter
    args.params = argparse.Namespace()#doesn't matter

    #########
    args.model = 'vae'#doesn't matter
    args.mode = 'test'#doesn't matter
    args.exp_name = 'review_anneal' #/dataset/exp_name
    args.description = 'different annealing strategies'
    args.question = ''
    args.extra_name = ''
    #########
    if sub_exp == 'yelp':
        args.dataset = 'yelp'

    if sub_exp == 'yahoo':
        args.label = False
        args.dataset = 'yahoo'
    args = default_text(args)#XXXmake sure that all grid things happen after this line

    args.warm_up = [30, 50, 100, 120]
    args.kl_start = 0.0
    args.kl_hold = 0
    return BaseExperiment(args)

# parser.add_argument('--warm_up', type=int, default=30)
# parser.add_argument('--kl_start', type=float, default=1.0)
# parser.add_argument('--kl_hold', type=int, default=0)
