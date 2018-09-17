import argparse
import torch
import numpy as np
# from experiments.reproduce import *
from experiments.beta_text import *
from experiments.beta_omni import *
# from my_paths import paths
# from saver.model_saver import ModelSaver
# from loggers.logger import TrainLogger
# from text import main
from image_beta import main as main_image
from text_beta import main as main_text
# from image_v import main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-resources', default=1, type=int, help="Number of resources to split jobs on")
    parser.add_argument('--resource-id',default=1, type=int, help="Resource ID from 1...#Resources" )
    parser.add_argument('--exp-names',default=None, type=str, help="Experiment Name" )
    cluster_config = parser.parse_args()
    return cluster_config

def run(cluster_config):
    '''Sets up multiple experiments
    example: python train.py --num-resources 1 --resource-id 1 --exp-name 'Exp1Name|SubExp1,Exp2Name|SubExp2'
    '''
    cc = cluster_config
    assert cc.resource_id is not None
    assert cc.num_resources is not None
    assert cc.resource_id <= cc.num_resources
    assert cc.exp_names != None

    exp_passed = cc.exp_names.split(',')
    print("Number Experiments:", len(exp_passed))

    all_exps_tups = []
    for combo_exp in exp_passed:
        (exp, sub_exp) = combo_exp.split('|')
        all_exps_tups.append((exp, sub_exp))
    run_experiment(cc, all_exps_tups)

def run_experiment(cluster_config, list_exps):
    all_jobs = []
    for (exp_name, sub_exp_name) in list_exps:
        exper = eval(exp_name)(sub_exp_name)
        all_jobs.extend(exper.get_jobs())
    setup(cluster_config, all_jobs)

def setup(cluster, configs):
    print("Running {} Jobs on {} resources".format(len(configs), cluster.num_resources))
    for i, task in enumerate(configs):
        if i % cluster.num_resources == (cluster.resource_id-1):
            if task.dataset in ['yahoo', 'synthetic', 'ptb', 'yelp']:
                main_text(task)
            elif task.dataset == 'omniglot':
                main_image(task)
            print(f"Task: {i} Done")

if __name__ == "__main__":
    # to run a Basic experiment
    # python train.py --num-resources 1 --resource-id 1 --exp-name 'vanilla|'
    # python train.py --num-resources 1 --resource-id 1 --exp-name 'best|'
    job_config = parse_args()
    run(job_config)