import torch
import torch.nn.functional as F

def gaussian_vae_loss(x, recon_mu, recon_logvar, mu, logvar):
    """
    Computes the negative ELBO loss for a VAE with Gaussian likelihood and prior.
    Args:
        x: original images, shape (batch, 1, 28, 28)
        recon_mu: mean of reconstructed images, shape (batch, 1, 28, 28)
        recon_logvar: log-variance of reconstructed images, shape (batch, 1, 28, 28)
        mu: mean of approximate posterior, shape (batch, latent_dim)
        logvar: log-variance of approximate posterior, shape (batch, latent_dim)
    Returns:
        loss: scalar tensor, mean negative ELBO over the batch
    """
    # Flatten images for pixel-wise computation
    x = x.view(x.size(0), -1)
    recon_mu = recon_mu.view(x.size(0), -1)
    recon_logvar = recon_logvar.view(x.size(0), -1)

    # Reconstruction loss: log N(x | mu, sigma^2)
    recon_loss = 0.5 * (recon_logvar + ((x - recon_mu) ** 2) / recon_logvar.exp())
    recon_loss = recon_loss.sum(dim=1)  # sum over features
    recon_loss = recon_loss.mean()      # mean over batch

    # KL divergence between q(z|x) and p(z) (both diagonal Gaussians)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl_div.mean()

    return recon_loss + kl_div


def beta_vae_loss(x, alpha, beta, mu, logvar, eps=1e-6):
    """
    Beta VAE loss (negative ELBO) using Beta likelihood.
    Args:
        x: input image in [0, 1], shape (batch, 1, 28, 28)
        alpha, beta: decoder outputs, shape (batch, 1, 28, 28)
        mu, logvar: encoder outputs, shape (batch, latent_dim)
    Returns:
        scalar tensor loss
    """
    # 0 ⟨ x ⟨ 1  ➊
    x = x.clamp(eps, 1.0 - eps)

    # flatten to (batch, D)
    x   = x.view(x.size(0), -1)
    a   = alpha.view(x.size(0), -1) + eps     # make sure >0
    b   = beta.view(x.size(0), -1)  + eps

    # log-normaliser  log B(a,b)
    log_B = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    # per-example log-likelihood  log p(x|z)
    log_px_z = ((a - 1) * torch.log(x) +
                (b - 1) * torch.log(1.0 - x) -
                log_B).sum(dim=1)            # sum over pixels

    # reconstruction part of –ELBO
    recon_term = -log_px_z.mean()

    # KL between q(z|x) and p(z) (same as Gaussian task)
    kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    return recon_term + kl_term
