import sys
import os
import time
import importlib
import argparse
import numpy as np

import torch
import torch.utils.data
from torch import nn, optim

from modules import ResNetEncoder, PixelCNNDecoder
from modules import VAE

clip_grad = 5.0
decay_epoch = 20
lr_decay = 0.5
max_decay = 5

def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', choices=['omniglot'], required=True, help='dataset to use')

    # optimization parameters
    parser.add_argument('--conv_nstep', type=int, default=20,
                         help='number of steps for inference work')
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

    id_ = "%s_burn%s_convs%d_ns%d_kls%.1f_warm%d_%d_%d" % \
            (args.dataset, args.burn, args.conv_nstep, args.nsamples,
             args.kl_start, args.warm_up, args.jobid, args.taskid)

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

    return args

def test(model, test_loader, mode, args):
    if args.eval == 1:
        f = open(args.save_path[:-2]+'_log_test', 'a')
    else:
        f = open(args.save_path[:-2]+'_log_val', 'a')
    report_kl_loss = report_rec_loss = 0
    report_num_examples = 0
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

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_examples

    nll = (report_kl_loss + report_rec_loss) / report_num_examples
    kl = report_kl_loss / report_num_examples

    f.write('%s --- avg_loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f\n' % \
               (mode, test_loss, report_kl_loss / report_num_examples,
                report_rec_loss / report_num_examples, nll))
    f.flush()
    print('%s --- avg_loss: %.4f, kl: %.4f, recon: %.4f, nll: %.4f' % \
           (mode, test_loss, report_kl_loss / report_num_examples,
            report_rec_loss / report_num_examples, nll))
    sys.stdout.flush()

    return test_loss, nll, kl

def calc_iwnll(model, test_loader, args):
    if args.eval == 1:
        f = open(args.save_path[:-2]+'_log_test', 'a')
    else:
        f = open(args.save_path[:-2]+'_log_val', 'a')
    report_nll_loss = 0
    report_num_examples = 0
    for id_, datum in enumerate(test_loader):
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size

        # TODO(junxian): check if __len__ function returns the number of examples
        if id_ % (round(len(test_loader) / 10)) == 0:
            print('iw nll computing %d0%%' % (id_/(round(len(test_loader) / 10))))
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_examples

    print('iw nll: %.4f' % nll)
    f.write('iw nll: %.4f\n' % nll)
    f.flush()
    sys.stdout.flush()

def burn_logic(args, epoch, kl_now=None, kl_prev=None):
    if args.burn == 0:
        return False
    elif epoch >= args.burn_from:
        if kl_now > kl_prev and args.burn_flag == False:
            args.burn_from += 1
            return False
        else:
            return True

    return False

def make_savepath(args):
    save_dir = "models/%s" % args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "%s_burn%s_from%d_convs%d_ns%d_kls%.1f_warm%d_%d_%d" % \
        (args.dataset, args.burn, args.burn_from, args.conv_nstep, args.nsamples,
         args.kl_start, args.warm_up, args.jobid, args.taskid)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path

def main(args):
    if args.save_path == '':
        make_savepath(args)

    if args.cuda:
        print('using cuda')

    if args.eval == 1:
        f = open(args.save_path[:-2]+'_log_test', 'a')
    else:
        f = open(args.save_path[:-2]+'_log_val', 'a')

    print(args)
    f.write(str(args)+'\n')
    f.flush()

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    all_data = torch.load(args.data_file)
    x_train, x_val, x_test = all_data

    # x_train = x_train[:500,:,:,:]
    # x_val = x_val[:500,:,:,:]
    # x_test = x_test[:500,:,:,:]
    # print(x_train.size())
    # print(x_val.size())
    # print(x_test.size())
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
    f.write('Train data: %d batches\n' % len(train_loader))
    f.write('Val data: %d batches\n' % len(val_loader))
    f.write('Test data: %d batches\n' % len(test_loader))
    f.flush()
    sys.stdout.flush()

    # if args.model == 'autoreg':
    #     args.latent_feature_map = 0

    encoder = ResNetEncoder(args)
    decoder = PixelCNNDecoder(args)

    vae = VAE(encoder, decoder, args).to(device)

    if args.eval:
        print('begin evaluation')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)
        if args.load_path == '':
            args.load_path = args.save_path
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test(vae, test_loader, "TEST", args)
            calc_iwnll(vae, test_loader, args)

        return

    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    opt_dict['lr'] = 0.001

    iter_ = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0

    prev_kl = 0
    current_kl = 0

    decay_cnt = 0
    # burn_flag = False
    args.burn_flag = False

    # burn_flag = True
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = 1.0 / (args.warm_up * len(train_loader))

    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_examples = 0

        args.burn_flag = burn_logic(args, epoch, current_kl, prev_kl)
        print('EP{}, Burning{}, {}'.format(epoch, args.burn_flag, args.burn))
        f.write('Burning:{}, Left{}\n'.format(args.burn_flag, args.burn))
        f.flush()
        # if epoch >= args.burn:
        #     burn_flag = False

        for datum in train_loader:
            batch_data, _ = datum
            batch_data = torch.bernoulli(batch_data)
            batch_size = batch_data.size(0)

            report_num_examples += batch_size

            # kl_weight = 1.0
            kl_weight = min(1.0, kl_weight + anneal_rate)


            stuck_cnt = 0
            sub_best_loss = 1e3
            sub_iter = 0
            batch_data_enc = batch_data
            while args.burn_flag and sub_iter <= args.conv_nstep:

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                loss, loss_rc, loss_kl, mix_prob = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)
                # print(mix_prob[0])

                loss = loss.mean(dim=-1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

                enc_optimizer.step()

                id_ = np.random.choice(x_train.size(0), args.batch_size, replace=False)

                batch_data_enc = x_train[id_]

                # if loss.item() < sub_best_loss:
                #     sub_best_loss = loss.item()
                #     stuck_cnt = 0
                # else:
                #     stuck_cnt += 1
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

            if not args.burn_flag:
                enc_optimizer.step()

            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % args.log_niter == 0:
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_examples

                f.write('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                       'time elapsed %.2fs\n' %
                       (epoch, iter_, train_loss, report_kl_loss / report_num_examples,
                       report_rec_loss / report_num_examples, time.time() - start))
                f.flush()
                print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                       'time elapsed %.2fs' %
                       (epoch, iter_, train_loss, report_kl_loss / report_num_examples,
                       report_rec_loss / report_num_examples, time.time() - start))
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_examples = 0

            iter_ += 1

        print('kl weight %.4f' % kl_weight)
        print('epoch: %d, VAL' % epoch)
        f.write('kl weight %.4f' % kl_weight)
        f.write('epoch: %d, VAL' % epoch)
        f.flush()
        vae.eval()

        with torch.no_grad():
            loss, nll, kl = test(vae, val_loader, "VAL", args)

        prev_kl = current_kl
        current_kl = kl
        if args.burn_flag:
            args.burn -= 1

        if loss < best_loss:
            print('update best loss')
            f.write('update best loss:{}\n'.format(loss))
            f.flush()
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
                f.write('new lr: %f\n' % opt_dict["lr"])
                f.flush()
                enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"], betas=(0.9, 0.999))
                dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"], betas=(0.9, 0.999))
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
