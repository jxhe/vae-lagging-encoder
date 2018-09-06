import sys
import os
import math
import torch
import numpy as np

import torch
from torch.autograd import Variable


def hmc_trajectory(current_z, current_v, U, grad_U, epsilon, L=10):
    """This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf.

    Args:
        U: function to compute potential energy/minus log-density
        grad_U: function to compute gradients w.r.t. U
        epsilon: (adaptive) step size
        L: number of leap-frog steps
        current_z: current position
    """

    # as of `torch-0.3.0.post4`, there still is no proper scalar support
    eps = epsilon.view(-1, 1)
    z = current_z
    v = current_v - grad_U(z).mul(eps).mul_(.5)


    for i in range(1, L+1):
        # next_v_half = current_v - epsilon/2 * grad_U(current_z)
        z = z + v.mul_(eps)
        if i < L:
            v = v - grad_U(z).mul(eps) #??? why no .5 here

    v = v - grad_U(z).mul(eps).mul(.5)
    v = -v # this is not needed; only here to conform to the math

    return z.detach(), v.detach()


def accept_reject(current_z, current_v,
                  z, v,
                  epsilon,
                  accept_hist, hist_len,
                  U, K=lambda v: torch.sum(v * v, 1)):
    """Accept/reject based on Hamiltonians for current and propose.

    Args:
        current_z: position BEFORE leap-frog steps
        current_v: speed BEFORE leap-frog steps
        z: position AFTER leap-frog steps
        v: speed AFTER leap-frog steps
        epsilon: step size of leap-frog.
                (This is only needed for adaptive update)
        U: function to compute potential energy (MINUS log-density)
        K: function to compute kinetic energy (default: kinetic energy in physics w/ mass=1)
    """
    # mdtype = type(current_z.data)
    mdtype = current_z.type()
    # print("current_z", current_z)

    current_H = U(current_z) + K(current_v)
    prop_H = U(z) + K(v)

    prob = torch.exp(current_H - prop_H)

    uniform_sample = torch.rand(prob.size())
    uniform_sample = Variable(uniform_sample.type(mdtype))
    # print(uniform_sample.type(), prob.type())
    accept = (prob > uniform_sample)

    accept = (prob > uniform_sample).type(mdtype)
    z = z.mul(accept.view(-1, 1)) + current_z.mul(1. - accept.view(-1, 1))
    accept_hist = accept_hist.add(accept)

    criteria = (accept_hist / hist_len > 0.65).type(mdtype)#???
    adapt = 1.02 * criteria + 0.98 * (1. - criteria)#???
    epsilon = epsilon.mul(adapt).clamp(1e-4, .5)#???

    # clear previous history & save memory, similar to detach
    z = Variable(z.data, requires_grad=True)
    epsilon = Variable(epsilon.data)
    accept_hist = Variable(accept_hist.data)

    return z, epsilon, accept_hist
