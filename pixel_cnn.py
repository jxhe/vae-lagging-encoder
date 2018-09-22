import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
import torch.utils.data
from torchvision.utils import save_image
from torch import nn, optim

from modules import ResNetEncoderV2, PixelCNNDecoderV2
# from modules import ResNetEncoder, PixelCNNDecoder
from modules import VAE
from loggers.logger import Logger
from eval_ais.ais import ais_trajectory
from make_small_test_script import load_indices_omniglot

clip_grad = 5.0
decay_epoch = 20
lr_decay = 0.5
max_decay = 5

def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', choices=['omniglot'], required=True, help='dataset to use')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument('--sample_from', type=str, default='', help='load model and perform sampling')

    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "models/%s" % args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "Pixel_%s_%d_%d" % \
            (args.dataset, args.jobid, args.taskid)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params

    args = argparse.Namespace(**vars(args), **params)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    return args



def test(model, test_loader, mode, args):

    report_loss = 0
    report_num_examples = 0
    mutual_info = []
    for datum in test_loader:
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size
        # z = torch.zeros(batch_size, 1, args.nz)
        # z = z.to(args.device)
        z = None
        loss = model.reconstruct_error(batch_data, z)
        # loss, loss_rc, loss_kl, mix_prob = model.loss(batch_data, 1.0, nsamples=args.nsamples)
        loss = loss.sum()
        report_loss += loss.item()



    nll = (report_loss) / report_num_examples

    print('%s --- avg_loss: %.4f' % \
           (mode, nll))
    sys.stdout.flush()

    return nll


def make_savepath(args):
    save_dir = "models/{}/{}".format(args.dataset, args.exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "%s_burn%d_ns%d_kls%.1f_warm%d_seed_%d" % \
        (args.dataset, args.burn, args.nsamples,
         args.kl_start, args.warm_up, args.seed)

    save_path = os.path.join(save_dir, id_ + '.pt')
    args.save_path = save_path

    if args.eval == 1:
        # f = open(args.save_path[:-2]+'_log_test', 'a')
        log_path = os.path.join(save_dir, id_ + '_log_test' + args.extra_name)
    else:
        # f = open(args.save_path[:-2]+'_log_val', 'a')
        log_path = os.path.join(save_dir, id_ + '_log_val' + args.extra_name)
    sys.stdout = Logger(log_path)

    if args.load_path == '':
        args.load_path = args.save_path
    # sys.stdout = open(log_path, 'a')

def seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

def main(args):
    args.latent_feature_map = 0 #pixelcnn xxx only
    args.nz = 0
    if args.save_path == '':
        make_savepath(args)
        seed(args)

    if args.cuda:
        print('using cuda')

    print(args)
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    all_data = torch.load(args.data_file)
    x_train, x_val, x_test = all_data

    # x_train = x_train[:500]
    # x_val = x_val[:500]
    # x_test = x_test[:500]

    x_train = x_train.to(device)
    x_val = x_val.to(device)
    x_test = x_test.to(device)
    y_size = 1
    y_train = x_train.new_zeros(x_train.size(0), y_size)
    y_val = x_train.new_zeros(x_val.size(0), y_size)
    y_test = x_train.new_zeros(x_test.size(0), y_size)
    print(torch.__version__)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print('Train data: %d batches' % len(train_loader))
    print('Val data: %d batches' % len(val_loader))
    print('Test data: %d batches' % len(test_loader))
    sys.stdout.flush()

    # if args.model == 'autoreg':
    #     args.latent_feature_map = 0

    # encoder = ResNetEncoderV2(args)
    decoder = PixelCNNDecoderV2(args)
    decoder = decoder.to(device)


    if args.eval:
        print('begin evaluation')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)
        vae.load_state_dict(torch.load(args.load_path))
        small_test_indices = load_indices_omniglot()
        small_x_test = x_test[small_test_indices, :,:,:]
        small_x_test = small_x_test.to(device)
        small_y_test = x_train.new_zeros(small_x_test.size(0), y_size)
        small_test_data = torch.utils.data.TensorDataset(small_x_test, small_y_test)
        small_test_loader = torch.utils.data.DataLoader(small_test_data, batch_size=args.batch_size, shuffle=True)
        test_elbo_iw_ais_equal(vae, small_test_loader, args, device)
        vae.eval()
        with torch.no_grad():
            test(vae, test_loader, "TEST", args)
            calc_iwnll(vae, test_loader, args)

        return

    dec_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    opt_dict['lr'] = 0.001

    iter_ = 0
    best_loss = 1e4
    best_nll = 0
    decay_cnt =0
    decoder.train()
    start = time.time()

    for epoch in range(args.epochs):
        report_nll_loss = 0
        report_num_examples = 0
        for datum in train_loader:
            batch_data, _ = datum
            batch_data = torch.bernoulli(batch_data)
            batch_size = batch_data.size(0)

            report_num_examples += batch_size

            dec_optimizer.zero_grad()

            z = torch.zeros(batch_size, 1, args.nz)
            z = z.to(device)
            z=None
            loss = decoder.reconstruct_error(batch_data, z)
            report_nll_loss += loss.sum().item()
            loss = loss.squeeze(dim=1)
            loss = loss.mean(dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)

            dec_optimizer.step()

            if iter_ % args.log_niter == 0:
                train_loss = report_nll_loss / report_num_examples
                print('epoch: %d, iter: %d, avg_loss: %.4f, time elapsed %.2fs' %
                       (epoch, iter_, train_loss, time.time() - start))
                sys.stdout.flush()

                report_nll_loss = 0
                report_num_examples = 0

            iter_ += 1


        print('epoch: %d, VAL' % epoch)

        decoder.eval()

        with torch.no_grad():
            loss = test(decoder, val_loader, "VAL", args)

        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            torch.save(decoder.state_dict(), args.save_path)

        if loss > best_loss:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                decoder.load_state_dict(torch.load(args.save_path))
                decay_cnt += 1
                print('new lr: %f' % opt_dict["lr"])
                dec_optimizer = optim.Adam(decoder.parameters(), lr=opt_dict["lr"])
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss = test(decoder, test_loader, "TEST", args)

        decoder.train()

    # compute importance weighted estimate of log p(x)
    decoder.load_state_dict(torch.load(args.save_path))
    decoder.eval()
    with torch.no_grad():
        loss = test(decoder, test_loader, "TEST", args)



if __name__ == '__main__':
    args = init_config()
    main(args)
