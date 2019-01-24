import sys
import pickle
import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData

from modules import LSTMEncoder, LSTMDecoder
from modules import VAE
from modules import generate_grid

clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5

def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # optimization parameters
    parser.add_argument('--optim', type=str, default='sgd', help='')
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # plotting parameters
    parser.add_argument('--plot_mode', choices=['multiple', 'single'], default='multiple',
        help="multiple denotes plotting multiple points, single denotes potting single point, \
        both of which have corresponding figures in the paper")

    parser.add_argument('--zmin', type=float, default=-20.0,
        help="boundary to approximate mean of model posterior p(z|x)")
    parser.add_argument('--zmax', type=float, default=20.0,
        help="boundary to approximate mean of model posterior p(z|x)")
    parser.add_argument('--dz', type=float, default=0.1,
        help="granularity to approximate mean of model posterior p(z|x)")

    parser.add_argument('--num_plot', type=int, default=500,
        help='number of sampled points to be ploted')

    parser.add_argument('--plot_niter', type=int, default=200,
        help="plot every plot_niter iterations")


    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--kl_start', type=float, default=1.0)

    # inference parameters
    parser.add_argument('--aggressive', type=int, default=0,
        help='apply aggressive training when nonzero, reduce to vanilla VAE when aggressive is 0')

    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument('--save_plot_data', type=str, default='')


    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    args.dataset = "synthetic"
    if args.plot_mode == "single":
        args.num_plot = 50

    save_dir = "models/%s" % args.dataset
    plot_dir = "plot_data/%s" % args.plot_mode

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    args.plot_dir = plot_dir

    id_ = "%s_aggressive%d_kls%.2f_warm%d_%d_%d_%d" % \
            (args.dataset, args.aggressive, args.kl_start,
             args.warm_up, args.jobid, args.taskid, args.seed)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params

    args = argparse.Namespace(**vars(args), **params)

    args.nz = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args

def test(model, test_data_batch, mode, args):

    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size


        loss, loss_rc, loss_kl = model.loss(batch_data, 1.0, nsamples=args.nsamples)

        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()


        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()

    mutual_info = calc_mi(model, test_data_batch)

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
           (mode, test_loss, report_kl_loss / report_num_sents, mutual_info,
            report_rec_loss / report_num_sents, nll, ppl))
    sys.stdout.flush()

    return test_loss, nll, kl, ppl

def calc_iwnll(model, test_data_batch, args):

    report_nll_loss = 0
    report_num_words = report_num_sents = 0
    for id_, i in enumerate(np.random.permutation(len(test_data_batch))):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size
        if id_ % (round(len(test_data_batch) / 10)) == 0:
            print('iw nll computing %d0%%' % (id_/(round(len(test_data_batch) / 10))))

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
    sys.stdout.flush()

def calc_mi(model, test_data_batch):
    mi = 0
    num_examples = 0
    for batch_data in test_data_batch:
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi_q(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples


def plot_multiple(model, plot_data, grid_z,
                  iter_, args):

    plot_data, sents_len = plot_data
    plot_data_list = torch.chunk(plot_data, round(args.num_plot / args.batch_size))

    infer_posterior_mean = []
    report_loss_kl = report_mi = report_num_sample = 0
    for data in plot_data_list:
        report_loss_kl += model.KL(data).sum().item()
        report_num_sample += data.size(0)
        report_mi += model.calc_mi_q(data) * data.size(0)

        # [batch, 1]
        posterior_mean = model.calc_model_posterior_mean(data, grid_z)

        infer_mean = model.calc_infer_mean(data)

        infer_posterior_mean.append(torch.cat([posterior_mean, infer_mean], 1))

    # [*, 2]
    infer_posterior_mean = torch.cat(infer_posterior_mean, 0)
    save_path = os.path.join(args.plot_dir, 'aggr%d_iter%d_multiple.pickle' % (args.aggressive, iter_))
    save_data = {'posterior': infer_posterior_mean[:,0].cpu().numpy(),
                 'inference': infer_posterior_mean[:,1].cpu().numpy(),
                 'kl': report_loss_kl / report_num_sample,
                 'mi': report_mi / report_num_sample
                 }
    pickle.dump(save_data, open(save_path, 'wb'))

def plot_single(infer_mean, posterior_mean, args):

    # [batch, time]
    infer_mean = torch.cat(infer_mean, 1)
    posterior_mean = torch.cat(posterior_mean, 1)


    save_path = os.path.join(args.plot_dir, 'aggr%d_single.pickle' % args.aggressive)
    save_data = {'posterior': posterior_mean.cpu().numpy(),
                 'inference': infer_mean.cpu().numpy(),
                 }
    pickle.dump(save_data, open(save_path, 'wb'))



def main(args):

    class uniform_initializer(object):
        def __init__(self, stdv):
            self.stdv = stdv
        def __call__(self, tensor):
            nn.init.uniform_(tensor, -self.stdv, self.stdv)


    class xavier_normal_initializer(object):
        def __call__(self, tensor):
            nn.init.xavier_normal_(tensor)

    if args.cuda:
        print('using cuda')

    print(args)


    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    train_data = MonoTextData(args.train_data)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, vocab=vocab)
    test_data = MonoTextData(args.test_data, vocab=vocab)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()

    log_niter = (len(train_data)//args.batch_size)//10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
    args.enc_nh = args.dec_nh

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    vae = VAE(encoder, decoder, args).to(device)


    if args.optim == 'sgd':
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=1.0)
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=1.0)
        opt_dict['lr'] = 1.0
    else:
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001, betas=(0.9, 0.999))
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001, betas=(0.9, 0.999))
        opt_dict['lr'] = 0.001

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = -1
    aggressive_flag = True if args.aggressive else False
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))

    plot_data = train_data.data_sample(nsample=args.num_plot, device=device, batch_first=True)

    if args.plot_mode == 'multiple':
        grid_z = generate_grid(args.zmin, args.zmax, args.dz, device, ndim=1)
        plot_fn = plot_multiple

    elif args.plot_mode == 'single':
        grid_z = generate_grid(args.zmin, args.zmax, args.dz, device, ndim=1)
        plot_fn = plot_single
        posterior_mean = []
        infer_mean = []

        posterior_mean.append(vae.calc_model_posterior_mean(plot_data[0], grid_z))
        infer_mean.append(vae.calc_infer_mean(plot_data[0]))

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)

    # plot_data_, _ = plot_data
    # train_data_batch = torch.chunk(plot_data_, round(args.num_plot / args.batch_size))

    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        for i in np.random.permutation(len(train_data_batch)):
            batch_data = train_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size

            report_num_sents += batch_size

            # kl_weight = 1.0
            kl_weight = min(1.0, kl_weight + anneal_rate)

            sub_iter = 1
            batch_data_enc = batch_data
            burn_num_words = 0
            burn_pre_loss = 1e4
            burn_cur_loss = 0
            while aggressive_flag and sub_iter < 100:

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                burn_batch_size, burn_sents_len = batch_data_enc.size()
                burn_num_words += (burn_sents_len - 1) * burn_batch_size

                loss, loss_rc, loss_kl = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)

                burn_cur_loss += loss.sum().item()
                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                enc_optimizer.step()

                id_ = np.random.random_integers(0, len(train_data_batch) - 1)

                batch_data_enc = train_data_batch[id_]

                if sub_iter % 15 == 0:
                    burn_cur_loss = burn_cur_loss / burn_num_words
                    if burn_pre_loss - burn_cur_loss < 0:
                        break
                    burn_pre_loss = burn_cur_loss
                    burn_cur_loss = burn_num_words = 0

                sub_iter += 1


            if args.plot_mode == 'single' and epoch == 0 and aggressive_flag:
                vae.eval()
                with torch.no_grad():
                    posterior_mean.append(posterior_mean[-1])
                    infer_mean.append(vae.calc_infer_mean(plot_data[0]))
                vae.train()


            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()


            loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)

            loss = loss.mean(dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            if not aggressive_flag:
                enc_optimizer.step()

            dec_optimizer.step()
            if args.plot_mode == 'single' and epoch == 0:
                vae.eval()
                with torch.no_grad():
                    posterior_mean.append(vae.calc_model_posterior_mean(plot_data[0], grid_z))

                    if aggressive_flag:
                        infer_mean.append(infer_mean[-1])
                    else:
                        infer_mean.append(vae.calc_infer_mean(plot_data[0]))
                vae.train()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % log_niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                if aggressive_flag or epoch == 0:
                    vae.eval()
                    mi = calc_mi(vae, val_data_batch)
                    vae.train()

                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                           report_rec_loss / report_num_sents, time.time() - start))
                else:
                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start))

                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            if iter_ % args.plot_niter == 0 and epoch == 0:
                vae.eval()
                with torch.no_grad():
                    if args.plot_mode == 'single' and iter_ != 0:
                        plot_fn(infer_mean, posterior_mean, args)
                        return
                    elif args.plot_mode == "multiple":
                        plot_fn(vae, plot_data, grid_z,
                                iter_, args)
                vae.train()

            iter_ += 1

            if aggressive_flag and (iter_ % len(train_data_batch)) == 0:
                vae.eval()
                cur_mi = calc_mi(vae, val_data_batch)
                vae.train()
                if cur_mi - pre_mi < 0:
                    aggressive_flag = False
                    print("STOP BURNING")

                pre_mi = cur_mi


                # return

        print('kl weight %.4f' % kl_weight)
        print('epoch: %d, VAL' % epoch)

        if args.plot_mode != '':
            with torch.no_grad():
                plot_fn(vae, plot_data, grid_z, iter_, args)

        vae.eval()
        with torch.no_grad():
            loss, nll, kl, ppl = test(vae, val_data_batch, "VAL", args)

        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            best_nll = nll
            best_kl = kl
            best_ppl = ppl
            torch.save(vae.state_dict(), args.save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >=15:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))
                print('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                if args.optim == 'sgd':
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"])
                    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"])
                else:
                    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
                    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, nll, kl, ppl = test(vae, test_data_batch, "TEST", args)

        vae.train()

    print('best_loss: %.4f, kl: %.4f, nll: %.4f, ppl: %.4f' \
          % (best_loss, best_kl, best_nll, best_ppl))

    sys.stdout.flush()


    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))
    vae.eval()
    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        calc_iwnll(vae, test_data_batch, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
