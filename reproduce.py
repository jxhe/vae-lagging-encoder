import argparse

from yahoo import main

args = argparse.Namespace()

args.anneal=True
args.batch_size=32
args.clip_grad=5.0
args.cuda=True
args.dec_dropout_in=0.5
args.dec_dropout_out=0.5
args.dz=0.1
args.epochs=75
args.eval=False
args.infer_steps=10
args.kl_start=0.1
args.load_model=''
args.lr=1.0
args.lr_decay=0.5
args.multi_infer=False
args.nepoch=1
args.nh=1024
args.ni=512
args.niter=200
args.nplot=10
args.nz=32
args.optim='sgd'
args.plot_niter=1
args.save_path='test'
args.seed=783435
args.server=''
args.stop_niter=20
args.test_data='/remote/bones/user/dspokoyn/text_gen/data/yahoo/yahoo.test.txt'
args.train_data='/remote/bones/user/dspokoyn/text_gen/data/yahoo/yahoo.train.txt'

args.enc_type = ''
args.enc_nh = 1024
args.dec_nh = 1024

args.warm_up=10
args.zmax=2.0
args.zmin=-2.0

main(args)