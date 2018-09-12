import torch

all_data = torch.load('../omniglot_data/omniglot.pt')
x_train, x_val, x_test = all_data
x_train_bern = torch.bernoulli(x_train)
x_val_bern = torch.bernoulli(x_val)
x_test_bern = torch.bernoulli(x_test)
torch.save([x_train_bern, x_val_bern, x_test_bern], 'omniglot_binarized.pt')
