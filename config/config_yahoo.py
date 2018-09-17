
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 1,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 200,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'yahoo_data/yahoo.train.txt',
    'val_data': 'yahoo_data/yahoo.valid.txt',
    'test_data': 'yahoo_data/yahoo.test.txt'
}
