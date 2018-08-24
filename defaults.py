
def fill_defaults(config):
    model_defaults(config)
    data_defaults(config)

def data_defaults(config):
    if config.dataset == 'yahoo': yahoo_default(config)
    elif config.dataset == 'ptb': ptb_default(config)

def model_defaults(config):
    if config.model == 'vae': vae_default(config)
    # elif config.model == 'vaeiter': vae_default(config)
    # elif config.model == 'vae_2lr': vae_default(config)

def yahoo_default(config):
    # args.nh=1024
    config.enc_nh = 1024
    config.dec_nh = 1024
    config.ni=512
    config.nz=32

    config.train_data='datasets/yahoo/data_yahoo_release/train.txt'
    config.test_data='datasets/yahoo/data_yahoo_release/valid.txt'
    # config.test_data='datasets/yahoo/data_yahoo_release/test.txt'
    config.save_path='test'

    pass

def ptb_default(config):
    config.enc_nh = 256
    config.dec_nh = 256
    config.ni=256
    config.nz=32
    config.train_data='datasets/yahoo/data_yahoo_release/train.txt'
    config.test_data='datasets/yahoo/data_yahoo_release/valid.txt'
    # config.test_data='datasets/yahoo/data_yahoo_release/test.txt'


def vae_default(config):
    config.cuda=True
    config.optim='sgd'
    config.clip_grad=5.0
    config.lr=1.0
    config.lr_decay=0.5
    config.decay_epoch = 5
    pass







