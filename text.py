import sys
import time
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData

from modules import LSTMEncoder, LSTMDecoder, MixLSTMEncoder
from modules import VAE, VisPlotter
from modules import generate_grid


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--nz', type=int, default=32, help='latent z size')
    parser.add_argument('--ni', type=int, default=512, help='word embedding size')
    parser.add_argument('--enc_nh', type=int, default=1024, help='LSTM hidden state size')
    parser.add_argument('--dec_nh', type=int, default=1024, help='LSTM hidden state size')
    parser.add_argument('--dec_dropout_in', type=float, default=0.5, help='LSTM decoder dropout')
    parser.add_argument('--dec_dropout_out', type=float, default=0.5, help='LSTM decoder dropout')
    parser.add_argument('--enc_type', type=str, default='lstm')

    # mix encoder parameters
    parser.add_argument('--kernel_num', type=int, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
        help='number of each kind of kernel')
    parser.add_argument('--mix_num', type=int, help='number of classes')
    parser.add_argument('--cnn_dropout', type=float, default=0.5)
    parser.add_argument('--baseline_path', type=str, help='path to load baseline model')

    # optimization parameters
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5)
    parser.add_argument('--clip_grad', type=float, default=5.0, help='')
    parser.add_argument('--optim', type=str, default='adam', help='')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--nsamples', type=int, default=3, help='number of samples')
    parser.add_argument('--iw_nsamples', type=int, default=10, help='number of samples')

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
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--zmin', type=float, default=-2.0)
    parser.add_argument('--zmax', type=float, default=2.0)
    parser.add_argument('--dz', type=float, default=0.1)
    parser.add_argument('--num_plot', type=int, default=10,
                         help='number of sampled points to be ploted')
    parser.add_argument('--plot_niter', type=int, default=1)
    parser.add_argument('--stop_niter', type=int, default=-1)
    parser.add_argument('--server', type=str, default='http://localhost')
    parser.add_argument('--env', type=str, default='main')


    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--kl_start', type=float, default=0.1)
    parser.add_argument('--anneal', action='store_true', default=False,
                         help='if perform annealing')

    # inference parameters
    parser.add_argument('--multi_infer', action='store_true', default=False,
                         help='if perform multiple steps of inference')
    parser.add_argument('--infer_steps', type=int, default=1,
                         help='number of inference steps performed each iteration')
    parser.add_argument('--burn', type=int, default=-1,
                         help='number of inference steps performed each iteration')

    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument('--save_path', type=str, default='', help='valid every nepoch epochs')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')


    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.kernel_sizes = [int(s) for s in args.kernel_sizes.split(',')]

    return args

def test(model, test_data_batch, args):

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

    print('avg_loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
           (test_loss, report_kl_loss / report_num_sents,
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

def plot_vae(plotter, model, plot_data, zrange,
             log_prior, iter_, num_slice):

    plot_data, sents_len = plot_data
    loss_kl = model.KL(plot_data).sum() / plot_data.size(0)
    posterior = model.eval_true_posterior_dist(plot_data, zrange, log_prior)
    inference = model.eval_inference_dist(plot_data, zrange)

    posterior_v = posterior.view(num_slice, num_slice)
    inference_v = inference.view(num_slice, num_slice)
    name = "iter %d, KL: %.4f" % (iter_, loss_kl.item())
    win_name = "iter %d" % iter_
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

    opt_dict = {"not_improved": 0, "lr": args.lr, "best_loss": 1e4}

    train_data = MonoTextData(args.train_data)

    vocab = train_data.vocab
    vocab_size = len(vocab)

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

    # if args.eval:
    #     print('begin evaluation')
    #     vae.load_state_dict(torch.load(args.load_model))
    #     vae.eval()
    #     calc_nll(hae, test_data, args)

    #     return

    if args.optim == 'sgd':
        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"])
        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"])
    else:
        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))

    if not args.multi_infer:
        args.infer_steps = 1

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

    if args.plot:
        layout=dict(dx=args.dz, dy=args.dz, x0=args.zmin, y0=args.zmin)
        plotter = VisPlotter(server=args.server, env=args.env, contour_layout=layout)
        plot_data = train_data.data_sample(nsample=args.num_plot, device=device, batch_first=True)
        zrange, num_slice = generate_grid(args.zmin, args.zmax, args.dz)
        zrange = zrange.to(device)
        log_prior = vae.eval_prior_dist(zrange)

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
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

            if epoch >= args.burn or args.burn < 0:
                args.infer_steps = 1

            for _ in range(args.infer_steps):

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()


                loss, loss_rc, loss_kl, mix_prob = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
                # print(mix_prob[0])

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()

                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), args.clip_grad)

                enc_optimizer.step()
                # assert (loss == loss).all()

            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % args.niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

                print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                       'time elapsed %.2fs' %
                       (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                       report_rec_loss / report_num_sents, time.time() - start))
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            if args.plot:
                if iter_ % args.plot_niter == 0:
                    with torch.no_grad():
                        plot_vae(plotter, vae, plot_data, zrange,
                                 log_prior, iter_, num_slice)
                # return

            iter_ += 1

            # if iter_ >= args.stop_niter and args.stop_niter > 0:
            #     return

        if epoch % args.nepoch == 0:
            print('kl weight %.4f' % kl_weight)
            print('epoch: %d, testing' % epoch)
            vae.eval()

            with torch.no_grad():
                loss, nll, kl, ppl = test(vae, test_data_batch, args)

            if loss < best_loss:
                print('update best loss')
                best_loss = loss
                best_nll = nll
                best_kl = kl
                best_ppl = ppl
                torch.save(vae.state_dict(), args.save_path)

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= args.decay_epoch:
                    opt_dict["best_loss"] = loss
                    opt_dict["not_improved"] = 0
                    opt_dict["lr"] = opt_dict["lr"] * args.lr_decay
                    print('new lr: %f' % opt_dict["lr"])
                    if args.optim == 'sgd':
                        enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"])
                        dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"])
                    else:
                        enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
                        dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"], betas=(0.5, 0.999))
            else:
                opt_dict["not_improved"] = 0
                opt_dict["best_loss"] = loss

            vae.train()

    print('best_loss: %.4f, kl: %.4f, nll: %.4f, ppl: %.4f' \
          % (best_loss, best_kl, best_nll, best_ppl))

    sys.stdout.flush()

    vae.eval()
    test_data_batch = test_data.create_data_batch(batch_size=2,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        calc_iwnll(vae, test_data_batch, args)

    # calc_nll(vae, test_data, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
