import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

from ctgan.synthesizers.dp_ctgan import CTGANSynthesizer, Generator, Discriminator
from ctgan.rdp_accountant import compute_rdp, get_privacy_spent
from sklearn.model_selection import train_test_split
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer


# Basic differential private CTGAN Synthesizer
class ADPCTGANSynthesizer(CTGANSynthesizer):
    """
        Optimized differential private CTGAN Synthesizer
    Args:
    k (int):
        Weight clustering. Defaults to 3.

    """
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4, betas=(0.5, 0.99),
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 clip_coeff=0.1, sigma=1, target_epsilon=3, target_delta=1e-5, k=3):

        super(CTGANSynthesizer, self).__init__(embedding_dim, generator_dim, discriminator_dim,
                         generator_lr, generator_decay, discriminator_lr, betas,
                         discriminator_decay, batch_size, discriminator_steps,
                         log_frequency, verbose, epochs, pac, cuda)

        self._clip_coeff = clip_coeff
        self._sigma = sigma
        self._target_epsilon = target_epsilon
        self._target_delta = target_delta
        self._k = k

        print(f'Init optimized CTGAN with differential privacy. '
              f'Target epsilon: {self._target_epsilon}')

    def get_config(self):
        return f"Clip Coefficient: {self._clip_coeff}\n" \
               f"Sigma: {self._sigma}\n" \
               f"Target Epsilon: {self._target_epsilon}\n" \
               f"Target delta: {self._target_delta}\n"


    def fit(self, train_data, discrete_columns=tuple()):

        d1, d2 = train_test_split(train_data, test_size=0.4)

        # Use CTGAN: train regular improved WGAN using the first dataset
        super().fit(d1, discrete_columns)

        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=self._betas,
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=self._betas, weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        i = 0
        self._G_losses = []
        self._D_losses = []
        self._epsilons = []
        epsilon = 0
        steps = 0
        epoch = 0

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        while epsilon < self._target_epsilon:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    pass

