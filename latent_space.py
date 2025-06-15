import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.colors as mcolors
from torch import Tensor
from sklearn.decomposition import PCA
import torchvision.utils as vutils

def plot_latent_space(
    encoder: nn.Module,
    test_x: Tensor,
    test_labels: Tensor,
    use_pca: bool = False,
    sample: bool = False,        # se True campiona, altrimenti usa μ

):
    """
    Visualize the 2D projection of latent space.
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    test_x = test_x.to(device)

    with torch.no_grad():
        mu, logvar = encoder(test_x)
        if sample:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu 

    values = z.cpu().numpy()
    plt.style.use("default")
    plt.figure(figsize=(12, 10))

    if use_pca:
        pca = PCA(n_components=2)
        values = pca.fit_transform(values)
        plt.xlabel("PC_1")
        plt.ylabel("PC_2")
    else:
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")

    c_map = plt.cm.viridis
    bounds = np.linspace(0, 9, 10)
    norm = mcolors.BoundaryNorm(bounds, c_map.N)
    
    sc = plt.scatter(values[:, 0], values[:, 1], c=test_labels.cpu(), cmap=c_map, norm=norm, s=10)
    plt.colorbar(sc)    
    plt.title("Latent Space Visualization")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def interpolate_latent_space(
    vae,
    test_x,
    test_labels,
    rows=8,               # quante coppie (righe) disegnare
    k=10,                 # quante interpolazioni per riga (colonne)
    img_size=28,
    device=None,
    sample=True,          # True = campiona z  |  False = usa mu
):
    """
    Genera una griglia che mostra l'interpolazione lineare nel latent space
    tra coppie di immagini con label differenti.

    - rows ........ numero di righe nella figura (coppie diverse)
    - k ............ numero di step di lambda in [0,1] (colonne)
    - sample ....... se True campiona  z = mu + sigma*eps , altrimenti usa mu

    Ritorna: figura matplotlib (visualizzata e salvata).
    """
    vae.eval()
    device = device or (next(vae.parameters()).device)
    test_x = test_x.to(device)

    # λ uniformemente distribuiti su [0,1]
    lambdas = torch.linspace(0.0, 1.0, steps=k, device=device)

    # buffer per tutte le immagini: (rows*k, 1, img_size, img_size)
    all_imgs = []

    with torch.no_grad():
        pairs_found = 0
        while pairs_found < rows:
            # 1) scegli due indici con label differente
            i, j = np.random.choice(len(test_x), 2, replace=False)
            if test_labels[i] == test_labels[j]:
                continue

            x_i = test_x[i : i + 1]  # shape (1,1,H,W)
            x_j = test_x[j : j + 1]

            # 2) ottieni z e z'
            if hasattr(vae, "encode"):                 # VAE con metodo encode
                z_i, _, _ = vae.encode(x_i)
                z_j, _, _ = vae.encode(x_j)
            else:                                      # VAE.forward restituisce mu,logvar
                _, mu_i, logvar_i, _ = vae(x_i)
                _, mu_j, logvar_j, _ = vae(x_j)
                if sample:
                    eps_i = torch.randn_like(mu_i)
                    eps_j = torch.randn_like(mu_j)
                    z_i = mu_i + eps_i * torch.exp(0.5 * logvar_i)
                    z_j = mu_j + eps_j * torch.exp(0.5 * logvar_j)
                else:
                    z_i = mu_i
                    z_j = mu_j

            # 3) interpolazione lineare per ogni λ
            for lam in lambdas:
                z_lam = lam * z_i + (1.0 - lam) * z_j   # shape (1, latent_dim)

                # 4) decodifica: generazione di x_λ
                decoder_out = vae.decoder(z_lam)

                if vae.output_dist == "gaussian":
                    mu_x, logvar_x = decoder_out
                    eps = torch.randn_like(mu_x)
                    x_hat = mu_x + eps * torch.exp(0.5 * logvar_x) if sample else mu_x
                else:  # beta‑VAE
                    alpha, beta = decoder_out
                    dist = torch.distributions.Beta(alpha, beta)
                    x_hat = dist.sample() if sample else alpha / (alpha + beta)

                all_imgs.append(x_hat.clamp(0, 1))

            pairs_found += 1

    # concatena le immagini e costruisci la griglia
    all_imgs_tensor = torch.cat(all_imgs, dim=0)        # (rows*k, 1, H, W)
    grid = vutils.make_grid(all_imgs_tensor, nrow=k, pad_value=1)

    # plotting
    plt.style.use("default")
    plt.figure(figsize=(k, rows))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Latent‑space interpolation (λ da 0→1 da sinistra a destra)")
    plt.tight_layout()
    plt.show()

    # opzionale: salva
    vutils.save_image(grid, "latent_interpolations.png")
    print("✔  Interpolazione salvata in latent_interpolations.png")