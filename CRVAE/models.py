import torch
import torch.nn as nn
import torch.nn.functional as F

import math


"""
This architecture is based on the WAE implementation found here:
https://github.com/1Konny/WAE-pytorch/
"""


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SimpleVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32, input_dim=32):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.activation = F.relu


    def sample(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu)
            std = logvar.exp()
            z = eps * std + mu
            return z
        else: 
            return mu


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z


    def encode(self, x):
        raise NotImplementedError


    def decode(self, mu, logvar):
        raise NotImplementedError



class CNNVAE(SimpleVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.input_dim in [32, 64, 128, 256]

        self.sq_dim = (self.input_dim // 32) * 2
        self.linear_dim = int(self.sq_dim*self.sq_dim*1024)

        self.cnn_enc_1 = nn.Conv2d(self.in_channels, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_enc_2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_enc_3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_enc_4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        self.linear_enc = nn.Linear(self.linear_dim, self.latent_dim*2)

        self.bn_enc_1 = nn.BatchNorm2d(128)
        self.bn_enc_2 = nn.BatchNorm2d(256)
        self.bn_enc_3 = nn.BatchNorm2d(512)
        self.bn_enc_4 = nn.BatchNorm2d(1024)


        self.linear_dec = nn.Linear(self.latent_dim, self.linear_dim)
        self.cnn_dec_1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_dec_2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_dec_3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn_dec_4 = nn.ConvTranspose2d(128, self.in_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn_dec_1 = nn.BatchNorm2d(512)
        self.bn_dec_2 = nn.BatchNorm2d(256)
        self.bn_dec_3 = nn.BatchNorm2d(128)
        

        self.apply(kaiming_init)


    def encode(self, x):
        out = self.activation(self.bn_enc_1(self.cnn_enc_1(x)))
        out = self.activation(self.bn_enc_2(self.cnn_enc_2(out)))
        out = self.activation(self.bn_enc_3(self.cnn_enc_3(out)))
        out = self.activation(self.bn_enc_4(self.cnn_enc_4(out)))

        out = out.view(-1, self.linear_dim)
        out = self.linear_enc(out)
        mu, logvar = out.chunk(2, dim=1)

        return mu, logvar 


    def decode(self, z):
        out = self.linear_dec(z)
        out = out.view(-1, 1024, self.sq_dim, self.sq_dim)

        out = self.activation(self.bn_dec_1(self.cnn_dec_1(out)))
        out = self.activation(self.bn_dec_2(self.cnn_dec_2(out)))
        out = self.activation(self.bn_dec_3(self.cnn_dec_3(out)))

        reconstruction = self.cnn_dec_4(out)
        return torch.sigmoid(reconstruction)


class MLPVAE(SimpleVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.linear_dim = self.input_dim  * self.input_dim * self.in_channels
        self.linear_enc_1 = nn.Linear(self.linear_dim, 1000)
        self.linear_enc_2 = nn.Linear(1000, 1000)
        self.linear_enc_3 = nn.Linear(1000, 1000)
        self.linear_enc_4 = nn.Linear(1000, self.latent_dim*2)

        self.linear_dec_1 = nn.Linear(self.latent_dim, 1000)
        self.linear_dec_2 = nn.Linear(1000, 1000)
        self.linear_dec_3 = nn.Linear(1000, 1000)
        self.linear_dec_4 = nn.Linear(1000, self.linear_dim)

        self.apply(kaiming_init)

    def encode(self, x):
        x = x.view(-1, self.linear_dim)
        out = self.activation(self.linear_enc_1(x))
        out = self.activation(self.linear_enc_2(out))
        out = self.activation(self.linear_enc_3(out))
        out = self.linear_enc_4(out)

        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        out = self.activation(self.linear_dec_1(z))
        out = self.activation(self.linear_dec_2(out))
        out = self.activation(self.linear_dec_3(out))
        out = self.linear_dec_4(out)

        reconstruction = out.view(-1, self.in_channels, self.input_dim, self.input_dim)
        return torch.sigmoid(reconstruction)
