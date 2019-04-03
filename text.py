import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData
from modules import VAE
from modules import LSTMEncoder, LSTMDecoder

clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')

    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')


    # decoding
    parser.add_argument('--decode_from', type=str, default="", help="pretrained model path")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument('--decode_input', type=str, default="", help="input text file to perform reconstruction")


    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")

    # inference parameters
    parser.add_argument('--aggressive', type=int, default=0,
                         help='apply aggressive training when nonzero, reduce to vanilla VAE when aggressive is 0')
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

    seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    args.seed = seed_set[args.taskid]

    id_ = "%s_aggressive%d_kls%.2f_warm%d_%d_%d_%d" % \
            (args.dataset, args.aggressive, args.kl_start,
             args.warm_up, args.jobid, args.taskid, args.seed)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path
    print("save path", args.save_path)

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    return args


def reconstruct(model, data, strategy, fname, device):
    with open(fname, "w") as fout:
        for batch_data, sent_len in data.data_iter(batch_size=1, device=device,
                                                   batch_first=True, shuffle=False):
            decoded_batch = model.reconstruct(batch_data, strategy)

            for sent in decoded_batch:
                fout.write(" ".join(sent) + "\n")

def sample_from_prior(model, z, strategy, fname):
    with open(fname, "w") as fout:
        decoded_batch = model.decode(z, strategy)

        for sent in decoded_batch:
            fout.write(" ".join(sent) + "\n")


def test(model, test_data_batch, mode, args, verbose=True):
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
    if verbose:
        print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
               (mode, test_loss, report_kl_loss / report_num_sents, mutual_info,
                report_rec_loss / report_num_sents, nll, ppl))
        sys.stdout.flush()

    return test_loss, nll, kl, ppl, mutual_info

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

def calc_au(model, test_data_batch, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item(), au_var


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

    train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()

    log_niter = (len(train_data)//args.batch_size)//10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.enc_type == 'lstm':
        encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args).to(device)

    if args.decode_from != "":
        print('begin decoding')
        vae.load_state_dict(torch.load(args.decode_from))
        vae.eval()
        save_dir = "samples/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = ".".join(args.decode_from.split("/")[-1].split(".")[:-1]) + \
                "_{}".format(args.decoding_strategy)
        with torch.no_grad():
            if args.decode_input != "":
                decode_data = MonoTextData(args.decode_input, vocab=vocab)

                reconstruct(vae, decode_data, args.decoding_strategy,
                    os.path.join(save_dir, path + ".rec"), args.device)
            else:
                z = vae.sample_from_prior(100)
                sample_from_prior(vae, z, args.decoding_strategy,
                    os.path.join(save_dir, path + ".sample"))

        return


    if args.eval:
        print('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)

            test(vae, test_data_batch, "TEST", args)
            au, au_var = calc_au(vae, test_data_batch)
            print("%d active units" % au)
            # print(au_var)

            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            calc_iwnll(vae, test_data_batch, args)

        return

    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=1.0, momentum=args.momentum)
    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=1.0, momentum=args.momentum)
    opt_dict['lr'] = 1.0

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = 0
    aggressive_flag = True if args.aggressive else False
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))

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

                # if sub_iter >= 30:
                #     break

            # print(sub_iter)

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

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % log_niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                if aggressive_flag or epoch == 0:
                    vae.eval()
                    with torch.no_grad():
                        mi = calc_mi(vae, val_data_batch)
                        au, _ = calc_au(vae, val_data_batch)
                    vae.train()

                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                           'au %d, time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                           report_rec_loss / report_num_sents, au, time.time() - start))
                else:
                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                           'time elapsed %.2fs' %
                           (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start))

                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            iter_ += 1

            if aggressive_flag and (iter_ % len(train_data_batch)) == 0:
                vae.eval()
                cur_mi = calc_mi(vae, val_data_batch)
                vae.train()
                print("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))
                if cur_mi - pre_mi < 0:
                    aggressive_flag = False
                    print("STOP BURNING")

                pre_mi = cur_mi

        print('kl weight %.4f' % kl_weight)

        vae.eval()
        with torch.no_grad():
            loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
            au, au_var = calc_au(vae, val_data_batch)
            print("%d active units" % au)
            # print(au_var)

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
                enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)
        au, au_var = calc_au(vae, test_data_batch)
        print("%d active units" % au)
        # print(au_var)

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        calc_iwnll(vae, test_data_batch, args)

if __name__ == '__main__':
    args = init_config()
    main(args)
