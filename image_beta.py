import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
import torch.utils.data
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

    # optimization parameters
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # beta-VAE
    parser.add_argument('--beta', type=float, default=1, help='beta-VAE')

    # inference parameters
    parser.add_argument('--burn', type=int, default=0,
                         help='number of epochs to performe multi-step update')

    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "models/%s" % args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "betaVAE_%s_burn%d_ns%d_%d_%d" % \
            (args.dataset, args.burn, args.nsamples,
             args.jobid, args.taskid)

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


def test_ais(model, test_loader, mode_split, args):
    for x in model.modules():
        x.eval()

    test_loss = 0
    report_num_examples = 0
    for datum in test_loader:
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size

        batch_ll = ais_trajectory(model, batch_data, mode='forward',
         prior=args.ais_prior, schedule=np.linspace(0., 1., args.ais_T),
          n_sample=args.ais_K, modality='image')
        test_loss += torch.sum(-batch_ll).item()


    nll = (test_loss) / report_num_examples

    print('%s AIS --- nll: %.4f' % \
           (mode_split, nll))
    sys.stdout.flush()
    return nll

def test_elbo_iw_ais_equal(vae, small_test_loader, args, device):
    nll_ais = test_ais(vae, small_test_loader, "SMALL%TEST", args)
    #########
    vae.eval()
    with torch.no_grad():
        loss_elbo, nll_elbo, kl_elbo = test(vae, small_test_loader, "SMALL%TEST", args)
    #########
    with torch.no_grad():
        nll_iw = calc_iwnll(vae, small_test_loader, args)
    #########
    #

    print('TEST: NLL Elbo:%.4f, IW:%.4f, AIS:%.4f'%(nll_elbo, nll_iw, nll_ais))



def test(model, test_loader, mode, args):

    report_kl_loss = report_rec_loss = 0
    report_num_examples = 0
    mutual_info = []
    for datum in test_loader:
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size


        loss, loss_rc, loss_kl, mix_prob = model.loss(batch_data, 1.0, nsamples=args.nsamples)
        # print(mix_prob)

        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()


        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()

    mutual_info = calc_mi(model, test_loader)

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_examples

    nll = (report_kl_loss + report_rec_loss) / report_num_examples
    kl = report_kl_loss / report_num_examples

    print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f' % \
           (mode, test_loss, report_kl_loss / report_num_examples, mutual_info,
            report_rec_loss / report_num_examples, nll))
    sys.stdout.flush()

    return test_loss, nll, kl

def calc_mi(model, test_loader):
    mi = 0
    num_examples = 0
    for datum in test_loader:
        batch_data, _ = datum
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi_q(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples

def calc_iwnll(model, test_loader, args):

    report_nll_loss = 0
    report_num_examples = 0
    for id_, datum in enumerate(test_loader):
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size

        if id_ % (round(len(test_loader) / 10)) == 0:
            print('iw nll computing %d0%%' % (id_/(round(len(test_loader) / 10))))
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_examples

    print('iw nll: %.4f' % nll)
    sys.stdout.flush()
    return nll

def make_savepath(args):
    save_dir = "models/{}/{}".format(args.dataset, args.exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "betaVAE_%s_burn%d_ns%d_seed_%d" % \
        (args.dataset, args.burn, args.nsamples,
         args.seed)

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

    encoder = ResNetEncoderV2(args)
    decoder = PixelCNNDecoderV2(args)

    vae = VAE(encoder, decoder, args).to(device)

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
        return
        vae.eval()
        with torch.no_grad():
            test(vae, test_loader, "TEST", args)
            calc_iwnll(vae, test_loader, args)

        return

    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001)
    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001)
    opt_dict['lr'] = 0.001

    iter_ = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    decay_cnt = pre_mi = best_mi = mi_not_improved =0
    burn_flag = True if args.burn else False
    vae.train()
    start = time.time()

    kl_weight = args.beta

    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_examples = 0
        for datum in train_loader:
            batch_data, _ = datum
            batch_data = torch.bernoulli(batch_data)
            batch_size = batch_data.size(0)

            report_num_examples += batch_size
            sub_iter = 1
            batch_data_enc = batch_data
            burn_num_examples = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            while burn_flag and sub_iter < 100:

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                burn_num_examples += batch_data_enc.size(0)
                loss, loss_rc, loss_kl, mix_prob = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)
                # print(mix_prob[0])

                burn_cur_loss += loss.sum().item()
                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                enc_optimizer.step()

                id_ = np.random.choice(x_train.size(0), args.batch_size, replace=False)

                batch_data_enc = torch.bernoulli(x_train[id_])

                if sub_iter % 10 == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_examples
                    if burn_pre_loss - burn_cur_loss < 0:
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_examples = 0

                sub_iter += 1

            # print(sub_iter)

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()


            loss, loss_rc, loss_kl, mix_prob = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)

            loss = loss.mean(dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            if not burn_flag:
                enc_optimizer.step()

            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % args.log_niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_examples
                if burn_flag or epoch == 0:
                    vae.eval()
                    with torch.no_grad():
                        mi = calc_mi(vae, val_loader)

                    vae.train()

                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_examples, mi,
                           report_rec_loss / report_num_examples, time.time() - start))
                else:
                     print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_examples,
                           report_rec_loss / report_num_examples, time.time() - start))
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_examples = 0

            iter_ += 1

            if burn_flag and (iter_ % len(train_loader)) == 0:
                vae.eval()
                cur_mi = calc_mi(vae, val_loader)
                vae.train()
                if cur_mi - best_mi < 0:
                    mi_not_improved += 1
                    if mi_not_improved == 5:
                        burn_flag = False
                        print("STOP BURNING")

                else:
                    best_mi = cur_mi

                pre_mi = cur_mi

        print('kl weight %.4f' % kl_weight)
        print('epoch: %d, VAL' % epoch)

        vae.eval()

        with torch.no_grad():
            loss, nll, kl = test(vae, val_loader, "VAL", args)

        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            best_nll = nll
            best_kl = kl
            torch.save(vae.state_dict(), args.save_path)

        if loss > best_loss:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))
                decay_cnt += 1
                print('new lr: %f' % opt_dict["lr"])
                enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"])
                dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"])
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, nll, kl = test(vae, test_loader, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))
    vae.eval()
    with torch.no_grad():
        loss, nll, kl = test(vae, test_loader, "TEST", args)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)

    with torch.no_grad():
        calc_iwnll(vae, test_loader, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
