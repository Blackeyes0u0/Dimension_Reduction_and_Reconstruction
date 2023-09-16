import torch
from torch import nn
import numpy as np

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # [b, 784] => [b, 20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )

        # [b, 20] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # flatten
        x = x.reshape(batch_size, 784)

        # encoder
        # [b, 20], consist of 10 mu and 10 sigma
        h_ = self.encoder(x)

        # [b, 20] => [b, 10] (mu) and [b, 10] (sigma)
        mu, sigma = h_.chunk(2, dim=1)

        # reparameterization trick
        h = mu + sigma * torch.randn_like(sigma)

        # kl divergence
        kld = 0.5 * torch.sum(mu ** 2 + sigma ** 2 - torch.log(1e-8 + sigma ** 2) - 1) / np.prod(x.shape)

        # decoder
        x = self.decoder(h)

        # reshape
        x = x.reshape(batch_size, 1, 28, 28)

        return x, kld