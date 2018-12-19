# VAE-lagging-encoder

This is PyTorch implementation of the paper:

```
Lagging Inference Networks and Posterior Collapse in Variational Autoencoders
Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick
ICLR 2019 (under review)
```

The code performs aggressive training of inference network to mitigate the issue of posterior collapse in VAE.

Please contact junxianh@cs.cmu.edu if you have any questions.

## Requirements

* Python 3
* PyTorch 0.4

## Data

Datasets used in the paper can be downloaded [here](https://drive.google.com/file/d/1jyeL4tXZHR4mJ1txdvs2yV2BKBO0KW4n/view?usp=sharing). For text experiments the data format is plain text where each line represents one training sample.

## Usage

Example script to train VAE on Yahoo dataset:
```
python text.py --dataset yahoo --aggressive 1 --warm_up 10 --kl_start 0.1
```

Here:

* `--aggressive` controls whether applies aggressive training or not
* `--kl_start` represents starting KL weight (set to 1.0 to disable KL annealing)
* `--warm_up` represents number of annealing epochs (KL weight increases from `kl_start` to 1.0 linearly in the first `warm_up` epochs)

To run the code on your own dataset, you need to create a new configuration file in `config` folder to specifiy network hyperparameters and datapath. If the new config file is `config/config_abc.py`, then `--dataset` needs to be set as `abc` accordingly.