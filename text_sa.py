import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable

from data import MonoTextData
from modules import VAE
from modules import LSTMEncoder, LSTMDecoder
from modules import generate_grid
from modules import OptimN2N
from eval_ais.ais import ais_trajectory
from loggers.logger import Logger

clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', choices=['yahoo', 'yelp', 'synthetic', 'ptb'], required=True, help='dataset to use')

    # optimization parameters
    parser.add_argument('--svi_steps', type=int, default=10)
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--kl_start', type=float, default=1.0)

    # parallel parameters
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids to use, only activated when ngpu > 1')

    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument('--train_from', type=str, default='')

    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    save_dir = "models/%s" % args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "%s_savae_ns%d_kls%.1f_warm%d_%d_%d" % \
            (args.dataset, args.nsamples,
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


def test(model, test_data_batch, meta_optimizer, mode, args, verbose=True):
    # for x in model.modules():
        # print(x.training)
        # x.eval() #not sure why this breaks???
    model.decoder.dropout_in.eval()
    model.decoder.dropout_out.eval()

    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size
        mean, logvar = model.encoder.forward(batch_data)
        var_params = torch.cat([mean, logvar], 1)
        mean_svi = Variable(mean.data, requires_grad=True)
        logvar_svi = Variable(logvar.data, requires_grad=True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], batch_data)

        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model.encoder.reparameterize(mean_svi_final, logvar_svi_final, 1)
        rc_svi = model.decoder.reconstruct_error(batch_data, z_samples).mean(dim=1)
        kl_svi = model.encoder.calc_kl(mean_svi_final, logvar_svi_final)

        report_rec_loss += rc_svi.sum().item()
        report_kl_loss += kl_svi.sum().item()

    mutual_info = calc_mi(model, test_data_batch)

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    if verbose:
        print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
               (mode, test_loss, report_kl_loss / report_num_sents, mutual_info,
                report_rec_loss / report_num_sents, nll, ppl))
        sys.stdout.flush()

    return test_loss, nll, kl, ppl, mutual_info

def calc_iwnll(model, test_data_batch, meta_optimizer, args, ns=100):
    model.decoder.dropout_in.eval()
    model.decoder.dropout_out.eval()

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

        loss = model.nll_iw(batch_data, meta_optimizer, nsamples=args.iw_nsamples, ns=ns)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
    sys.stdout.flush()
    return nll, ppl

def calc_mi(model, test_data_batch):
    mi = 0
    num_examples = 0
    for batch_data in test_data_batch:
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi_q(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples

def test_elbo_iw_ais_equal(vae, small_test_data, meta_optimizer, args, device):
    #### Compare ELBOvsIWvsAIS on Same Data
    small_test_data_batch = small_test_data.create_data_batch(batch_size=20,
                                                  device=device,
                                                  batch_first=True)
    ###ais###
    nll_ais, ppl_ais = test_ais(vae, small_test_data_batch, "10%TEST", args)
    #########
    vae.eval()
    with torch.no_grad():
        loss_elbo, nll_elbo, kl_elbo, ppl_elbo, mutual_info = test(vae, small_test_data_batch, meta_optimizer, "10%TEST", args)
    #########
    with torch.no_grad():
        nll_iw, ppl_iw = calc_iwnll(vae, small_test_data_batch, meta_optimizer, args, ns=20)
    #########
    print('TEST: NLL Elbo:%.4f, IW:%.4f, AIS:%.4f,\t Perp Elbo:%.4f,\tIW:%.4f,\tAIS:%.4f,\tMIT:%.4f'%(nll_elbo, nll_iw, nll_ais, ppl_elbo, ppl_iw, ppl_ais, mutual_info))

def make_savepath(args):
    save_dir = "models/{}/{}".format(args.dataset, args.exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "%s_savae_ns%d_kls%.1f_warm%d_seed_%d" % \
            (args.dataset, args.nsamples,
             args.kl_start, args.warm_up, args.seed)

    # id_ = "%s_savae_nref%d_kls%.1f_warm%d_seed_%d" % \
    #     (args.dataset, args.svi_steps,
    #      args.kl_start, args.warm_up, args.seed)

    save_path = os.path.join(save_dir, id_ + '.pt')
    args.save_path = save_path

    if args.eval == 1:
        # f = open(args.save_path[:-2]+'_log_test', 'a')
        log_path = os.path.join(save_dir, id_ + '_log_test')
    else:
        # f = open(args.save_path[:-2]+'_log_val', 'a')
        log_path = os.path.join(save_dir, id_ + '_log_val')
    sys.stdout = Logger(log_path)
    # sys.stdout = open(log_path, 'a')

def seed(args):
    if args.ngpu != 1:
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

def main(args):

    class uniform_initializer(object):
        def __init__(self, stdv):
            self.stdv = stdv
        def __call__(self, tensor):
            nn.init.uniform_(tensor, -self.stdv, self.stdv)


    class xavier_normal_initializer(object):
        def __call__(self, tensor):
            nn.init.xavier_normal_(tensor)
    if args.save_path == '':
        make_savepath(args)
        seed(args)

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

    encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
    args.enc_nh = args.dec_nh

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    vae = VAE(encoder, decoder, args).to(device)

    def variational_loss(input, x, vae, z = None):
        mean, logvar = input
        z_samples = vae.encoder.reparameterize(mean, logvar, 1, z)
        kl = vae.encoder.calc_kl(mean, logvar)
        reconstruct_err = vae.decoder.reconstruct_error(x, z_samples).squeeze(1)
        return (reconstruct_err + kl_weight*kl).mean(-1)

    update_params = list(vae.decoder.parameters())
    meta_optimizer = OptimN2N(variational_loss, vae, update_params, eps=1e-5,
                              lr=[1, 1],
                              iters=args.svi_steps, momentum=0.5,
                              acc_param_grads=1,
                              max_grad_norm=5)

    if args.eval:
        kl_weight = 1.0
        # small_test_data = MonoTextData(args.small_test_data, vocab=vocab)
        print('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))

        # test_elbo_iw_ais_equal(vae, small_test_data, meta_optimizer, args, device)
        test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                      device=device,
                                                      batch_first=True)
        test_data_batch = test_data_batch[:10]

        test(vae, test_data_batch, meta_optimizer, "TEST", args)


        test_data_batch = test_data.create_data_batch(batch_size=1,
                                                      device=device,
                                                      batch_first=True)
        calc_iwnll(vae, test_data_batch, meta_optimizer, args)

        return

    optimizer = optim.SGD(vae.parameters(), lr=1.0)
    opt_dict['lr'] = 1.0

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
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
    # xxx
    #train_data_batch = train_data_batch[:10]
    #val_data_batch = val_data_batch[:10]
    #test_data_batch = test_data_batch[:10]
    # xxx

    if args.train_from != '':
        vae.load_state_dict(torch.load(args.train_from))
        loss, nll, kl, ppl, _ = test(vae, val_data_batch, meta_optimizer, "VAL", args)
        best_loss = opt_dict["best_loss"] = loss
        torch.save(vae.state_dict(), args.save_path)
        vae.train()

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

            optimizer.zero_grad()
            mean, logvar = vae.encoder.forward(batch_data)
            var_params = torch.cat([mean, logvar], 1)
            mean_svi = Variable(mean.data, requires_grad=True)
            logvar_svi = Variable(logvar.data, requires_grad=True)
            var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], batch_data)

            mean_svi_final, logvar_svi_final = var_params_svi
            z_samples = vae.encoder.reparameterize(mean_svi_final, logvar_svi_final, 1)
            rc_svi = vae.decoder.reconstruct_error(batch_data, z_samples).mean(dim=1)
            kl_svi = vae.encoder.calc_kl(mean_svi_final, logvar_svi_final)

            var_loss = (rc_svi + kl_weight * kl_svi).mean(-1)
            var_loss.backward(retain_graph=True)
            var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad])
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            optimizer.step()

            report_rec_loss += rc_svi.sum().item()
            report_kl_loss += kl_svi.sum().item()

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

        loss, nll, kl, ppl, cur_mi = test(vae, val_data_batch, meta_optimizer, "VAL", args)
        # vae.eval()
        # with torch.no_grad():

        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            best_nll = nll
            best_kl = kl
            best_ppl = ppl
            torch.save(vae.state_dict(), args.save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))
                print('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                optimizer = optim.SGD(vae.parameters(), lr=opt_dict["lr"])
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            loss, nll, kl, ppl, _ = test(vae, test_data_batch, meta_optimizer, "TEST", args)
            # with torch.no_grad():

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    test(vae, test_data_batch, meta_optimizer, "TEST", args)
    # vae.eval()
    # with torch.no_grad():

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    # with torch.no_grad():
    calc_iwnll(vae, test_data_batch, meta_optimizer, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
