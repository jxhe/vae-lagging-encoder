import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from modules import ResNetEncoderV2, PixelCNNDecoderV2
from modules import VAE
from modules import OptimN2N
from loggers.logger import Logger

clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5

def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', choices=['omniglot'], required=True, help='dataset to use')

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

    id_ = "%s_savae_nref%d_kls%.1f_warm%d_%d_%d" % \
            (args.dataset, args.svi_steps,
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
        torch.backends.cudnn.deterministic = True

    return args

def test(vae, test_loader, meta_optimizer, mode, args):
    for x in vae.modules():
        x.eval()

    report_kl_loss = report_rec_loss = 0
    report_num_examples = 0
    mutual_info = []
    for datum in test_loader:
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size
        mean, logvar = vae.encoder.forward(batch_data)
        var_params = torch.cat([mean, logvar], 1)
        mean_svi = Variable(mean.data, requires_grad=True)
        logvar_svi = Variable(logvar.data, requires_grad=True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], batch_data)

        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = vae.encoder.reparameterize(mean_svi_final, logvar_svi_final, 1)
        rc_svi = vae.decoder.reconstruct_error(batch_data, z_samples).mean(dim=1)
        kl_svi = vae.encoder.calc_kl(mean_svi_final, logvar_svi_final)

        report_rec_loss += rc_svi.sum().item()
        report_kl_loss += kl_svi.sum().item()

    mutual_info = calc_mi(vae, test_loader)

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

def calc_iwnll(model, test_loader, meta_optimizer, args):

    report_nll_loss = 0
    report_num_examples = 0
    for id_, datum in enumerate(test_loader):
        batch_data, _ = datum
        batch_size = batch_data.size(0)

        report_num_examples += batch_size

        if id_ % (round(len(test_loader) / 10)) == 0:
            print('iw nll computing %d0%%' % (id_/(round(len(test_loader) / 10))))
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, meta_optimizer, nsamples=args.iw_nsamples)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_examples

    print('iw nll: %.4f' % nll)
    sys.stdout.flush()

def make_savepath(args):
    save_dir = "models/{}/{}".format(args.dataset, args.exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_ = "%s_savae_nref%d_kls%.1f_warm%d_seed_%d" % \
        (args.dataset, args.svi_steps,
         args.kl_start, args.warm_up, args.seed)

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
    # xxx
    # x_train = x_train[:500,:,:,:]
    # x_val = x_val[:500,:,:,:]
    # x_test = x_test[:500,:,:,:]
    # xxx
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

    def variational_loss(input, img, vae, z = None):
        mean, logvar = input
        z_samples = vae.encoder.reparameterize(mean, logvar, 1, z)
        kl = vae.encoder.calc_kl(mean, logvar)
        reconstruct_err = vae.decoder.reconstruct_error(img, z_samples).squeeze(1)
        return (reconstruct_err + kl_weight*kl).mean(-1)

    update_params = list(vae.decoder.parameters())
    meta_optimizer = OptimN2N(variational_loss, vae, update_params, eps=1e-5,
                              lr=[1, 1],
                              iters=args.svi_steps, momentum=0.5,
                              acc_param_grads=1,
                              max_grad_norm=5)

    if args.eval:
        print('begin evaluation')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)
        vae.load_state_dict(torch.load(args.load_path))
        test(vae, test_loader, meta_optimizer, "TEST", args)
        calc_iwnll(vae, test_loader, meta_optimizer, args)
        # vae.eval()
        # with torch.no_grad():

        return

    optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))
    opt_dict['lr'] = 0.001

    iter_ = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    decay_cnt = 0
    burn_flag = True
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = 1.0 / (args.warm_up * len(train_loader))

    if args.train_from != '':
        vae.load_state_dict(args.train_from)
        test(vae, val_loader, meta_optimizer, "VAL", args)
        vae.train()

    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_examples = 0
        for datum in train_loader:
            batch_data, _ = datum
            batch_data = torch.bernoulli(batch_data)
            batch_size = batch_data.size(0)

            report_num_examples += batch_size

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
                train_loss = (report_rec_loss  + report_kl_loss) / report_num_examples
                # vae.eval()
                # with torch.no_grad():
                #     mi = calc_mi(vae, val_loader)

                # vae.train()

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


        loss, nll, kl = test(vae, val_loader, meta_optimizer, "VAL", args)
        # vae.eval()
        # with torch.no_grad():

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
                optimizer = optim.Adam(vae.parameters(), lr=opt_dict["lr"], betas=(0.9, 0.999))
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            # with torch.no_grad():
            loss, nll, kl = test(vae, test_loader, meta_optimizer, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))
    loss, nll, kl = test(vae, test_loader, meta_optimizer, "TEST", args)
    # with torch.no_grad():

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    calc_iwnll(vae, test_loader, meta_optimizer, args)
    # vae.eval()
    # with torch.no_grad():

if __name__ == '__main__':
    args = init_config()
    main(args)
