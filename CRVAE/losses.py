import torch
import torch.nn.functional as F


def elbo_loss(mu, logvar, recon, original, beta):
    batch_size = original.size(0)
    BCE = F.binary_cross_entropy(
            recon.view(batch_size, -1), original.view(batch_size, -1), reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    vae_loss = BCE + beta * KLD
    return vae_loss, BCE, KLD 


def cr_loss(mu, logvar, mu_aug, logvar_aug, gamma):
    """
    distance between two gaussians
    """
    std_orig = logvar.exp()
    std_aug = logvar_aug.exp()

    cr_loss = 0.5 * torch.sum(2 * torch.log(std_orig / std_aug) - \
            1 + (std_aug ** 2 + (mu_aug - mu_orig) ** 2) / std_orig ** 2,
            dim=1).mean()

    cr_loss *= gamma
    return cr_loss
