import torch
import torch.nn as nn
import torch.nn.functional as F

from CRVAE.losses import *


class CRVAE(object):
    def __init__(self, gamma=1e-3, beta_1=1, beta_2=1):
        """
        gamma: is equivalent to the \lambda parameter Equation 3 
        in https://arxiv.org/abs/2105.14859 (defaults to 1e-3)
        beta_1: parameter for the VAE loss to reconstruct the original 
        samples (defaults to 1)  
        beta_2: parameter for the VAE loss to reconstruct the augmented
        samples (defaults to 1)  
        """
        super().__init__()

        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def calculate_loss(self, model, original_samples, augmented_samples):
        mu, logvar, recon = model(original_samples)
        mu_aug, logvar_aug, recon_aug = model(augmented_samples)

        vae_loss, bce, kld = elbo_loss(
            mu, 
            logvar, 
            recon,
            original_samples, 
            self.beta1
        )

        aug_vae_loss, aug_bce, aug_kld = elbo_loss(
            mu_aug, 
            logvar_aug, 
            recon_aug,
            augmented_samples, 
            self.beta2
        )

        cr_vae_loss = cr_loss(mu, logvar, mu_aug, logvar_aug, self.gamma)


        loss = cr_var_loss + vae_loss + aug_vae_loss

        log = {}
        log['loss'] = loss
        log['vae_loss'] = vae_loss
        log['aug_vae_loss'] = aug_vae_loss
        log['cr_vae_loss'] = cr_vae_loss
        log['bce'] = bce
        log['kld'] = kld
        log['aug_bce'] = bce
        log['aug_kld'] = kld

        return loss, log
        
        
