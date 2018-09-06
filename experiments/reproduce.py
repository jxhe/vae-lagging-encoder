import copy
from bluetorch.experiment.base_experiment import BaseExperiment, BaseAnalysis
import argparse
from defaults import fill_defaults


def vanilla(sub_exp=None):
    print(sub_exp, "sub_exp")
    config = argparse.Namespace()
    config.options = argparse.Namespace()
    config.params = argparse.Namespace()

    #########
    config.model = 'vae'
    config.mode = 'test'
    config.dataset = 'yahoo'
    config.exp_name = 'trial'
    config.description = 'Vanilla VAE Baseline'
    config.question = ''
    config.extra_name = ''
    #########
    fill_defaults(config)
    ###########
    config.niter=200
    config.nplot=10
    config.plot_niter=1
    config.zmax=None
    config.zmin=None
    config.plot = False
    config.dz=0.1
    config.server=''
    config.load_model=''
    #
    config.batch_size=32

    config.enc_type = 'lstm'
    config.nsamples = 3

    config.dec_dropout_in=0.5
    config.dec_dropout_out=0.5
    config.nepoch=1
    config.eval=False

    config.anneal=True
    config.kl_start=.1
    config.warm_up=10

    config.infer_steps=0
    config.multi_infer=False
    config.stop_niter=None
    config.burn = -1

    config.iw_nsamples = 500
    config.eval_batches = 10
    config.ais_prior = 'normal'
    config.ais_T = 100
    config.ais_K = 5

    config.epochs=2

    return BaseExperiment(config)

def best(sub_exp=None):
    print(sub_exp, "sub_exp")
    config = argparse.Namespace()
    config.options = argparse.Namespace()
    config.params = argparse.Namespace()

    #########
    config.model = 'vae'
    config.mode = 'test'
    config.dataset = 'yahoo'
    config.exp_name = 'trial'
    config.description = 'Best VAE'
    config.question = ''
    config.extra_name = ''
    #########
    fill_defaults(config)
    ###########
    config.niter=200
    config.nplot=10
    config.plot_niter=1
    config.zmax=None
    config.zmin=None
    config.plot = False
    config.dz=0.1
    config.server=''
    config.load_model=''
    #
    config.batch_size=32

    config.enc_type = 'lstm'
    config.nsamples = 3

    config.dec_dropout_in=0.5
    config.dec_dropout_out=0.5
    config.nepoch=1
    config.eval=False

    config.anneal=True
    config.kl_start= .1
    config.warm_up=10

    config.multi_infer=True
    config.infer_steps=10
    config.stop_niter=None
    config.burn = 1
    config.iw_nsamples = 10

    config.epochs=50

    return BaseExperiment(config)



