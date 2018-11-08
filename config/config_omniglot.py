
params={
    'img_size': [1,28,28],
    'nz': 32,
    'enc_layers': [64, 64, 64],
    'dec_kernel_size': [9,9,9,7,7,7,5,5,5,3,3,3],
    'dec_layers': [32,32,32,32,32,32,32,32,32,32,32,32],
    'latent_feature_map': 4,
    'batch_size': 50,
    'epochs': 1000,
    'log_niter': 10,
    'test_nepoch': 5,
    'data_file': 'omniglot_data/omniglot.pt'
}

params_v2={
    'img_size': [[1,28,28]],
    'nz': 32,
    'enc_layers': [[64, 64, 64]],
    'dec_kernel_size': [[9,9,9,7,7,7,5,5,5,3,3,3]],
    'dec_layers': [[32,32,32,32,32,32,32,32,32,32,32,32]],
    'latent_feature_map': 4,
    'batch_size': 50,
    'epochs': 1000,
    'log_niter': 100,
    'test_nepoch': 5,
    'data_file': 'omniglot_data/omniglot.pt'
}
