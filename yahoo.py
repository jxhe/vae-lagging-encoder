import sys
import time
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData

from modules import VarLSTMEncoder, VarLSTMDecoder
from modules import VAE, VisPlotter
from modules import generate_grid


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--nz', type=int, default=32, help='latent z size')
    parser.add_argument('--ni', type=int, default=512, help='word embedding size')
    parser.add_argument('--nh', type=int, default=1024, help='LSTM hidden state size')
    parser.add_argument('--dec_dropout_in', type=float, default=0.5, help='LSTM decoder dropout')
    parser.add_argument('--dec_dropout_out', type=float, default=0.5, help='LSTM decoder dropout')

    # optimization parameters
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='')
    parser.add_argument('--optim', type=str, default='adam', help='')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')


    # KL annealing parameters
    # parser.add_argument('--warm_up', type=int, default=5, help='')
    # parser.add_argument('--kl_start', type=float, default=0.1, help='')

    # data parameters
    parser.add_argument('--train_data', type=str, default='datasets/yahoo/data_yahoo_release/train.txt',
                        help='training data file')
    parser.add_argument('--test_data', type=str, default='datasets/yahoo/data_yahoo_release/test.txt',
                        help='testing data file')

    # log parameters
    parser.add_argument('--niter', type=int, default=50, help='report every niter iterations')
    parser.add_argument('--nepoch', type=int, default=1, help='valid every nepoch epochs')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_model', type=str, default='')

    # plot parameters
    parser.add_argument('--zmin', type=float, default=-2.0)
    parser.add_argument('--zmax', type=float, default=2.0)
    parser.add_argument('--dz', type=float, default=0.1)
    parser.add_argument('--nplot', type=int, default=10,
                         help='number of sampled points to be ploted')
    parser.add_argument('--plot_niter', type=int, default=1)
    parser.add_argument('--stop_niter', type=int, default=-1)
    parser.add_argument('--server', type=str, default='http://localhost')


    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--kl_start', type=float, default=0.1)
    parser.add_argument('--anneal', action='store_true', default=False,
                         help='if perform annealing')


    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument('--save_path', type=str, default='', help='valid every nepoch epochs')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')


    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args

def test(model, test_data, args):

    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    for batch_data, sents_len in test_data.data_iter(batch_size=args.batch_size,
                                                     device=args.device,
                                                     batch_first=True):

        batch_size = len(batch_data)

        report_num_sents += batch_size

        # sents_len counts both the start and end symbols
        sents_len = torch.LongTensor(sents_len)

        # not predict start symbol
        report_num_words += (sents_len - 1).sum().item()


        loss_rc, loss_kl = model.loss((batch_data, sents_len), nsamples=1)

        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()


        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    print('avg_loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
           (test_loss, report_kl_loss / report_num_sents,
            report_rec_loss / report_num_sents, nll, ppl))
    sys.stdout.flush()

    return test_loss, nll, kl, ppl

def plot_vae(plotter, model, plot_data, zrange,
             log_prior, iter_, num_slice):

    plot_data, sents_len = plot_data
    loss_kl = model.KL(plot_data)
    posterior = model.eval_true_posterior_dist(plot_data, zrange, log_prior)
    inference = model.eval_inference_dist(plot_data, zrange)

    for i, posterior_ in enumerate(posterior):
        posterior_v = posterior_.view(num_slice, num_slice)
        inference_v = inference[i].view(num_slice, num_slice)
        name = "iter %d, posterior of sample %d, KL: %.4f" % \
                (iter_, i, loss_kl[i].item())
        win_name = "iter %d, sample%d" % (iter_, i)
        plotter.plot_contour([posterior_v, inference_v], win=win_name, name=name)



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

    print('model saving path: %s' % args.save_path)

    print(args)

    schedule = args.epochs / 5
    lr_ = args.lr

    train_data = MonoTextData(args.train_data)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    test_data = MonoTextData(args.test_data, vocab)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    sys.stdout.flush()

    model_init = xavier_normal_initializer()
    emb_init = uniform_initializer(0.1)

    encoder = VarLSTMEncoder(args, vocab_size, model_init, emb_init)
    decoder = VarLSTMDecoder(args, vocab, model_init, emb_init)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    vae = VAE(encoder, decoder, args).to(device)

    # if args.eval:
    #     print('begin evaluation')
    #     vae.load_state_dict(torch.load(args.load_model))
    #     vae.eval()
    #     calc_nll(hae, test_data, args)

    #     return

    if args.optim == 'sgd':


        optimizer = optim.SGD(vae.parameters(), lr=lr_)
    else:
        optimizer = optim.Adam(vae.parameters(), lr=lr_, betas=(0.5, 0.999))

    iter_ = 0
    decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    vae.train()
    start = time.time()

    if args.anneal:
        kl_weight = args.kl_start
    else:
        kl_weight = 1.0

    anneal_rate = 1.0 / (args.warm_up * (len(train_data) / args.batch_size))

    # calc_nll(hae, test_data, args)
    # layout=dict(dx=args.dz, dy=args.dz, x0=args.zmin, y0=args.zmin)
    # plotter = VisPlotter(server=args.server, contour_layout=layout)
    # plot_data = train_data.data_sample(nsample=args.nplot, device=device, batch_first=True)
    # zrange, num_slice = generate_grid(args.zmin, args.zmax, args.dz)
    # zrange = zrange.to(device)
    # log_prior = vae.eval_prior_dist(zrange)

    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        for batch_data, sents_len in train_data.data_iter(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True):

            batch_size = len(batch_data)
            # sents_len counts both the start and end symbols
            sents_len = torch.LongTensor(sents_len)

            # not predict start symbol
            report_num_words += (sents_len - 1).sum().item()

            report_num_sents += batch_size

            optimizer.zero_grad()

            loss_rc, loss_kl = vae.loss((batch_data, sents_len), nsamples=1)
            #print('-----------------')
            #print(loss_bce.mean().data)
            #print(loss_kl.mean().data)
            #print(hlg.mean().data)
            # assert (loss_bce == loss_bce).all()
            # assert (loss_kl == loss_kl).all()
            # assert (hlg == hlg).all()
            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            kl_weight = min(1.0, kl_weight + anneal_rate)
            # kl_weight = 1.0

            loss = (loss_rc + kl_weight * loss_kl) / batch_size

            # assert (loss == loss).all()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)
            optimizer.step()

            if iter_ % args.niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

                print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                       'time elapsed %.2fs' %
                       (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                       report_rec_loss / report_num_sents, time.time() - start))
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0


            # if iter_ % args.plot_niter == 0:
            #     with torch.no_grad():
            #         plot_vae(plotter, vae, plot_data, zrange,
            #                  log_prior, iter_, num_slice)
            #     # return

            iter_ += 1

            # if iter_ >= args.stop_niter and args.stop_niter > 0:
            #     return

        if epoch % args.nepoch == 0:
            print('kl weight %.4f' % kl_weight)
            print('epoch: %d, testing' % epoch)
            vae.eval()

            with torch.no_grad():
                loss, nll, kl, ppl = test(vae, test_data, args)

            if loss < best_loss:
                print('update best loss')
                best_loss = loss
                best_nll = nll
                best_kl = kl
                best_ppl = ppl
                torch.save(vae.state_dict(), args.save_path)

            vae.train()

        if (epoch + 1) % schedule == 0:
            print('update lr, old lr: %f' % lr_)
            lr_ = lr_ * args.lr_decay
            print('new lr: %f' % lr_)
            if args.optim == 'sgd':
                optimizer = optim.SGD(vae.parameters(), lr=lr_)
            else:
                optimizer = optim.Adam(vae.parameters(), lr=lr_, betas=(0.5, 0.999))

    print('best_loss: %.4f, kl: %.4f, nll: %.4f, ppl: %.4f' \
          % (best_loss, best_kl, best_nll, best_ppl))
    sys.stdout.flush()

    # vae.eval()
    # calc_nll(vae, test_data, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
