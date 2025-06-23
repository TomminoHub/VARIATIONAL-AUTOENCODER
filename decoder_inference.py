"""
Helper code for task 4:
1. Reconstruct an imagewithout using the encoder by directly
   optimising a latent vector.
2. Complete  the right half of an image the same way.
"""

import torch
import torch.nn.functional as F


def beta_negative_log_likelihood(x_true, alpha, beta, epsilon=1e-6):
    """
    Pixel-wise −log p(x|z) for the Beta decoder.
    """
    x_clamped = x_true.clamp(epsilon, 1.0 - epsilon)
    log_B = (torch.lgamma(alpha) + torch.lgamma(beta)
             - torch.lgamma(alpha + beta))
    nll = -((alpha - 1) * torch.log(x_clamped)
            + (beta - 1) * torch.log(1.0 - x_clamped)
            - log_B)
    return nll.view(nll.size(0), -1).sum(dim=1)



@torch.no_grad()
def decoder_mean_image(decoder_output, output_distribution):
    """
    Given whatever the decoder returns, turn it into a deterministic
    [0,1] image that is easy to visualise.
    """
    alpha, beta = decoder_output
    return alpha / (alpha + beta)  # mean of Beta(α,β)



def optimise_latent_vector(
    model,
    target_image,
    steps=8000,
    learning_rate=5e-2,
    prior_weight=1e-4
):
    """
    Find a latent vector z* that makes the decoder's output look like
    `target_image`.   The encoder is not used.
    Returns (reconstructed_image, best_z).
    """
    device = target_image.device
    latent_vector = torch.zeros(
        1, model.latent_dim, device=device, requires_grad=True
    )

    optimiser = torch.optim.Adam([latent_vector], lr=learning_rate)

    for i in range(steps):
        optimiser.zero_grad()

        decoder_output = model.decoder(latent_vector)

 # beta
        alpha, beta = decoder_output
        neg_log_likelihood = beta_negative_log_likelihood(
            target_image, alpha, beta
        )

        # −log p(z)  for a standard normal prior N(0,I)
        neg_log_prior = 0.5 * latent_vector.pow(2).sum()

        loss = neg_log_likelihood + prior_weight * neg_log_prior
        loss.backward()
        optimiser.step()

    # decode one last time (no gradients needed now)
    with torch.no_grad():
        reconstructed = decoder_mean_image(
            model.decoder(latent_vector), model.output_dist
        )

    return reconstructed.squeeze(0), latent_vector.detach().squeeze(0)


def inpaint_right_half(
    model,
    left_half_image,
    steps=8000,
    learning_rate=5e-2,
    prior_weight=1e-4
):
    """
    Given an image where the right half is not present, infer a latent vector
    using the left half, then have the decoder fill in the right.
    Returns (completed_image, best_z).
    """
    device = left_half_image.device
    latent_vector = torch.zeros(
        1, model.latent_dim, device=device, requires_grad=True
    )
    optimiser = torch.optim.Adam([latent_vector], lr=learning_rate)

    # build a binary mask: 1 on the left half, 0 on the right
    mask = torch.ones_like(left_half_image)
    mask[..., :, left_half_image.shape[-1] // 2 :] = 0

    for _ in range(steps):
        optimiser.zero_grad()
        decoder_output = model.decoder(latent_vector)

        alpha, beta = decoder_output
        neg_log_likelihood = beta_negative_log_likelihood(
            left_half_image * mask, alpha, beta
        )

        neg_log_prior = 0.5 * latent_vector.pow(2).sum()
        loss = neg_log_likelihood + prior_weight * neg_log_prior
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        completed_image = decoder_mean_image(
            model.decoder(latent_vector), model.output_dist
        )

    return completed_image.squeeze(0), latent_vector.detach().squeeze(0)
