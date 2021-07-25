import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from ctgan.rdp_accountant import compute_rdp, get_privacy_spent


class ADPCTGANSynthesizer(DPCTGANSynthesizer):
    """
        Advanced dp-CTGAN
        With optimization strategies
        Algorithm: https://arxiv.org/pdf/1801.01594.pdf

    Args:

        k (int):


    """
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 private=False, clip_coeff=0.1, sigma=1, target_epsilon=3, target_delta=1e-5,
                 weight_cluster= None, k=None):
        assert batch_size % 2 == 0

        super(ADPCTGANSynthesizer, self).__init__(embedding_dim, generator_dim, discriminator_dim,
                         generator_lr, generator_decay, discriminator_lr,
                         discriminator_decay, batch_size, discriminator_steps,
                         log_frequency, verbose, epochs, pac, cuda,
                         private, clip_coeff, sigma, target_epsilon, target_delta)

        self._weight_cluster = weight_cluster
        self._k = k

    def fit(self, train_data, discrete_columns=tuple()):

        pass
