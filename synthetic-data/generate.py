import argparse

import torch
import torch.nn as nn

from lm_lstm import LM

def init_config():
    parser = argparse.ArgumentParser(description='generate synthetic data')

    # Model parameters.
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',default=false,
                        help='use CUDA')
    parser.add_argument('--ni', type=int, default=100,
                        help='input dimension')
    parser.add_argument('--nz', type=int, default=2,
                        help='latent z dimension')
    parser.add_argument('--nh', type=int, default=100,
                        help='number of hidden units')
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nsamples', type=int, default=10000, 
                        help='number of generated samples')
    parser.add_argument('--length', type=int, default=10,
                        help='length of each generated sentence')
    parser.add_argument('--outpath', type=int, )


    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    return args


def main(args):
    
    def uniform_initializer(stdv):
        def forward(tensor):
            nn.init.uniform_(tensor, -stdv, stdv)


    fout = open(args.outpath, 'w')
    model = LM(args, uniform_initializer)

    for i in range(round(args.nsamples / args.batch_size)):
        samples = model.sample(args.batch_size, args.length)
        for sent in samples:
            for wid in sent:
                fout.write('%d ' % wid)
            fout.write('\n')

    fout.close()

if __name__ == '__main__':
    args = init_config()
    main(args)

