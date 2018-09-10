import numpy as np
import time

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
# from utils import log_normal, log_bernoulli, log_mean_exp, discretized_logistic, safe_repeat
from eval_ais.hmc import hmc_trajectory, accept_reject
from tqdm import tqdm


def ais_trajectory(model, batch_data, mode='forward', prior='inference', schedule=np.linspace(0., 1., 500), n_sample=100, modality=None):
    """Compute annealed importance sampling trajectories for a batch of data.
    Could be used for *both* forward and reverse chain in bidirectional Monte Carlo
    (default: forward chain with linear schedule).

    Args:
        model (vae.VAE): VAE model
        loader (iterator): iterator that returns pairs, with first component being `x`,
            second would be `z` or label (will not be used)
        mode (string): indicate forward/backward chain; must be either `forward` or 'backward'
        prior: 'inference' or 'standard'. inference uses q(z|x) or draws z~p(z)
        schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`;
            foward chain has increasing values, whereas backward has decreasing values
        n_sample (int): number of importance samples (i.e. number of parallel chains
            for each datapoint)

    Returns:
        A list where each element is a torch.autograd.Variable that contains the
        log importance weights for a single batch of data
    """

    assert mode == 'forward' or mode == 'backward', 'Should have either forward/backward mode'

    def log_f_i_normal(z, batch_data, t):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        zeros = Variable(torch.zeros(B, z_size).to(device))
        log_prior = log_normal(z, zeros, zeros)
        z = z.unsqueeze(1).expand(B, 1, z_size)
        log_likelihood = -model.decoder.reconstruct_error(batch_data, z).squeeze(dim=1)
        return log_prior + log_likelihood.mul_(t)

    # def log_f_i_inference(z, batch_i, batch_o, t):
    #     zeros = Variable(torch.zeros(B, z_size).to(device))
    #     log_inference = log_normal(z, q_mu, q_logvar)
    #     log_prior = log_normal(z, zeros, zeros)
    #     _, log_likelihood = model.forward_decode_batch(z, batch_i, batch_o, keep_rate=1.0) #??? keeprate

    #     return log_inference.mul_(1-t) + log_prior.mul_(t) + log_likelihood.mul_(t)

    if prior == 'inference':
        log_f_i = log_f_i_inference
    elif prior == 'normal':
        log_f_i = log_f_i_normal

    # shorter aliases
    z_size = model.args.nz
    device = model.args.device

    _time = time.time()
    logws = []  # for output

    print ('In %s mode' % mode)

    if modality == 'image':
        batch_size = batch_data.size(0)
    elif modality == 'text':
        batch_size, sent_len = batch_data.size()

    B = batch_size * n_sample
    batch_data = safe_repeat(batch_data, n_sample)

    # batch of step sizes, one for each chain
    epsilon = Variable(torch.ones(B).to(device)).mul_(0.01)
    # accept/reject history for tuning step size
    accept_hist = Variable(torch.zeros(B).to(device))
    # record log importance weight; volatile=True reduces memory greatly
    logw = torch.zeros(B).to(device)

    # initial sample of z
    if mode == 'forward':
        if prior == 'normal':
            current_z = Variable(torch.randn(B, z_size).to(device), requires_grad=True)
        elif prior == 'inference':
        # post_z = post_z.contiguous()
            current_z = safe_repeat(post_z, n_sample)
    else:
        #implement the reverse direction correctly
        fooooohere
        current_z = Variable(safe_repeat(post_z, n_sample).to(device), requires_grad=True)

    for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
        # update log importance weight
        log_int_1 = log_f_i(current_z, batch_data, t0)
        log_int_2 = log_f_i(current_z, batch_data, t1)
        logw.add_(log_int_2.detach() - log_int_1.detach())
        # resample speed
        current_v = Variable(torch.randn(current_z.size()).to(device))

        def U(z):
            return -log_f_i(z, batch_data, t1)

        def grad_U(z):
            # grad w.r.t. outputs; mandatory in this case
            grad_outputs = torch.ones(B).to(device)
                # torch.autograd.grad default returns volatile
            grad = torchgrad(U(z), z, grad_outputs=grad_outputs, retain_graph=False, create_graph=False)[0]
                # clip by norm
            grad = torch.clamp(grad, -B*z_size*100, B*z_size*100)
            # needs variable wrapper to make differentiable
            grad = Variable(grad.data, requires_grad=True)
            return grad

        def normalized_kinetic(v):
            zeros = Variable(torch.zeros(B, z_size).to(device))
            # this is superior to the unnormalized version
            return -log_normal(v, zeros, zeros)

        z, v = hmc_trajectory(current_z, current_v, U, grad_U, epsilon)

        # accept-reject step
        current_z, epsilon, accept_hist = accept_reject(current_z, current_v,
                                                        z, v,
                                                        epsilon,
                                                        accept_hist, j,
                                                        U, K=normalized_kinetic)

    # IWAE lower bound
    # print("logw", logw.size())
    logw = log_mean_exp(logw.view(n_sample, -1).transpose(0, 1)) #sum over K samples
    if mode == 'backward':
        logw = -logw
    # print("logw", logw.size())
    return logw.data

    # this if this was over multiple batches############
    # logws.append(logw.data)
    # print ('Time elapse %.4f, last batch stats %.4f' % (time.time()-_time, logw.mean().cpu().data.numpy()))
    # _time = time.time()
    # return logws
    ####################################################

def log_normal(x, mean, logvar):
    #TODO MAKE SURE UNDERSTAND THIS
    """Implementation WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out.
    Args:
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """
    # question about the logvar in denuminator
    return -0.5 * (logvar.sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))


def log_normal_full_cov(x, mean, L):
    #TODO MAKE SURE UNDERSTAND THIS
    """Log density of full covariance multivariate Gaussian.
    Note: results are off by the constant log(), since this
    quantity cancels out in p(z) and q(z|x)."""

    def batch_diag(M):
        diag = [t.diag() for t in torch.functional.unbind(M)]
        diag = torch.functional.stack(diag)
        return diag

    def batch_inverse(M, damp=False, eps=1e-6):
        damp_matrix = Variable(torch.eye(M[0].size(0)).type(M.data.type())).mul_(eps)
        inverse = []
        for t in torch.functional.unbind(M):
            # damping to ensure invertible due to float inaccuracy
            # this problem is very UNLIKELY when using double
            m = t if not damp else t + damp_matrix
            inverse.append(m.inverse())
        inverse = torch.functional.stack(inverse)
        return inverse

    L_diag = batch_diag(L)
    term1 = -torch.log(L_diag).sum(1)

    L_inverse = batch_inverse(L)
    scaled_diff = L_inverse.matmul((x - mean).unsqueeze(2)).squeeze()
    term2 = -0.5 * (scaled_diff ** 2).sum(1)

    return term1 + term2


def log_bernoulli(logit, target):
    """
    Args:
        logit:  [B, X]
        target: [B, X]

    Returns:
        output:      [B]
    """

    loss = -F.relu(logit) + torch.mul(target, logit) - torch.log(1. + torch.exp( -logit.abs() ))
    loss = torch.sum(loss, 1)

    return loss


def mean_squared_error(prediction, target):

    prediction, target = flatten(prediction), flatten(target)
    diff = prediction - target

    return -torch.sum(torch.mul(diff, diff), 1)


def discretized_logistic(mu, logs, x):
    """Probability mass follow discretized logistic.
    https://arxiv.org/pdf/1606.04934.pdf. Assuming pixel values scaled to be
    within [0,1]. """

    sigmoid = torch.nn.Sigmoid()

    s = torch.exp(logs).unsqueeze(-1).unsqueeze(-1)
    logp = torch.log(sigmoid((x + 1./256. - mu) / s) - sigmoid((x - mu) / s) + 1e-7)

    return logp.sum(-1).sum(-1).sum(-1)


def log_mean_exp(x):

    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


def numpy_nan_guard(arr):
    return np.all(arr == arr)


def safe_repeat(x, n):
    return x.repeat(n, *[1 for _ in range(len(x.size()) - 1)])
