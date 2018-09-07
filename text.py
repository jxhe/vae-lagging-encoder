import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData
# from modules import VAE, VisPlotter
from modules import VAE
from modules import LSTMEncoder, LSTMDecoder
from modules import generate_grid
from eval_ais.ais import ais_trajectory

clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', choices=['yahoo', 'synthetic', 'ptb'], required=True, help='dataset to use')

    # optimization parameters
    parser.add_argument('--optim', type=str, default='sgd', help='')

    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--conv_nstep', type=int, default=20,
                         help='number of steps of not improving loss to determine convergence, only used when burning is turned on')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--kl_start', type=float, default=1.0)

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

    id_ = "%s_optim%s_burn%s_convs%d_constlen_ns%d_kls%.1f_warm%d_%d_%d" % \
            (args.dataset, args.optim, args.burn, args.conv_nstep, args.nsamples,
             args.kl_start, args.warm_up, args.jobid, args.taskid)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path
    print("save path", args.save_path)

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params

    args = argparse.Namespace(**vars(args), **params)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args

def test_ais(model, test_data_batch, mode_split, args):
    model.decoder.dropout_in.eval()
    model.decoder.dropout_out.eval()

    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    test_loss = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        print("WOOT", batch_size, sent_len)
        batch_ll = ais_trajectory(model, batch_data, mode='forward', prior=args.ais_prior, schedule=np.linspace(0., 1., args.ais_T), n_sample=args.ais_K)
        test_loss += torch.sum(-batch_ll).item()

    # test_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

    nll = (test_loss) / report_num_sents

    ppl = np.exp(nll * report_num_sents / report_num_words)
    print("SENTS, WORDS", report_num_sents, report_num_words)
    print('%s AIS --- nll: %.4f, ppl: %.4f' % \
           (mode_split, nll, ppl))
    sys.stdout.flush()
    return nll, ppl


def test(model, test_data_batch, mode, args):
    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size


        loss, loss_rc, loss_kl, mix_prob = model.loss(batch_data, 1.0, nsamples=args.nsamples)
        # print(mix_prob)

        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()


        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    # print("SENTS, WORDS", report_num_sents, report_num_words)
    print('%s --- avg_loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
           (mode, test_loss, report_kl_loss / report_num_sents,
            report_rec_loss / report_num_sents, nll, ppl))
    sys.stdout.flush()

    return test_loss, nll, kl, ppl

def calc_iwnll(model, test_data_batch, args, ns=100):
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
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples, ns=ns)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    print("SENTS, WORDS", report_num_sents, report_num_words)

    print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
    sys.stdout.flush()
    return nll, ppl

def test_elbo_iw_ais_equal(vae, small_test_data, args, device):
    #### Compare ELBOvsIWvsAIS on Same Data
    small_test_data_batch = small_test_data.create_data_batch(batch_size=20,
                                                  device=device,
                                                  batch_first=True)
    ###ais###
    nll_ais, ppl_ais = test_ais(vae, small_test_data_batch, "10%TEST", args)
    #########
    vae.eval()
    with torch.no_grad():
        loss_elbo, nll_elbo, kl_elbo, ppl_elbo = test(vae, small_test_data_batch, "10%TEST", args)
    #########
    with torch.no_grad():
        nll_iw, ppl_iw = calc_iwnll(vae, small_test_data_batch, args, ns=20)
    #########
    print('TEST: NLL Elbo:%.4f, IW:%.4f, AIS:%.4f,\t Perp Elbo:%.4f,\tIW:%.4f,\tAIS:%.4f'%(nll_elbo, nll_iw, nll_ais, ppl_elbo, ppl_iw, ppl_ais))


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
    # logger = TrainLogger(args, paths)

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

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    if args.enc_type == 'lstm':
        encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    elif args.enc_type == 'mix':
        encoder = MixLSTMEncoder(args, vocab_size, model_init, emb_init)
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    vae = VAE(encoder, decoder, args).to(device)

    if args.eval:
        small_test_data = MonoTextData(args.small_test_data, vocab=vocab)
        print('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))

        test_elbo_iw_ais_equal(vae, small_test_data, args, device)

        test_data_batch = test_data.create_data_batch(batch_size=1,
                                                      device=device,
                                                      batch_first=True)
        vae.eval()
        with torch.no_grad():
            calc_iwnll(vae, test_data_batch, args)

        return

    if args.optim == 'sgd':
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=1.0, momentum=args.momentum)
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=1.0, momentum=args.momentum)
        opt_dict['lr'] = 1.0
    else:
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_dict['lr'] = 0.001

    iter_ = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    burn_flag = True
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = 1.0 / (args.warm_up * (len(train_data) / args.batch_size))

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)

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

            if epoch >= args.burn:
                burn_flag = False

            stuck_cnt = 0
            sub_best_loss = 1e3
            sub_iter = 0
            batch_data_enc = batch_data
            while burn_flag and sub_iter <= args.conv_nstep:

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                loss, loss_rc, loss_kl, mix_prob = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)
                # print(mix_prob[0])

                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                enc_optimizer.step()

                id_ = np.random.random_integers(0, len(train_data_batch) - 1)

                batch_data_enc = train_data_batch[id_]

                if loss.item() < sub_best_loss:
                    sub_best_loss = loss.item()
                    stuck_cnt = 0
                else:
                    stuck_cnt += 1
                sub_iter += 1

                # if sub_iter >= 30:
                #     break

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
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

                print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                       'time elapsed %.2fs' %
                       (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                       report_rec_loss / report_num_sents, time.time() - start))
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            iter_ += 1

        print('kl weight %.4f' % kl_weight)

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
                if args.optim == 'sgd':
                    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                else:
                    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
                    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, nll, kl, ppl = test(vae, test_data_batch, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl = test(vae, test_data_batch, "TEST", args)

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        calc_iwnll(vae, test_data_batch, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
