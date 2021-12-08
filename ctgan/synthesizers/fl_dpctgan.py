import torch

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Generator, Discriminator
from ctgan.synthesizers.dp_ctgan import DPCTGANSynthesizer
from matplotlib import pyplot as plt
import os
import datetime
from torch import optim
import warnings


# Basic differential private CTGAN Synthesizer
class FLDPCTGANSynthesizer(DPCTGANSynthesizer):
    """
        Basic differential private CTGAN Synthesizer
        Algorithm: https://arxiv.org/pdf/1801.01594.pdf
    Args:
        private (bool):
            Inject random noise during optimization procedure in order to achieve
            differential privacy. Currently only naively inject noise.
            Defaults to ``False``.
        clip_coeff (float):
            Gradient clipping bound. Defaults to ``0.1``.
        sigma (int):
            Noise scale. Defaults to ``2``.
        epsilon (int):
            Differential privacy budget
        delta (float):
            Differential privacy budget

    """
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4, betas=(0.5, 0.99),
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 clip_coeff=0.1, sigma=1, target_epsilon=3, target_delta=1e-5,
                 train_data=None, discrete_columns=()):

        super(FLDPCTGANSynthesizer, self).__init__(embedding_dim, generator_dim, discriminator_dim,
                         generator_lr, generator_decay, discriminator_lr, betas,
                         discriminator_decay, batch_size, discriminator_steps,
                         log_frequency, verbose, epochs, pac, cuda,
                         clip_coeff, sigma, target_epsilon, target_delta)

        self.train_data = train_data
        self.discrete_columns = discrete_columns

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

