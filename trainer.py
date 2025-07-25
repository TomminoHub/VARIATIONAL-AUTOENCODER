import os
import torch
import torch.optim as optim
import yaml
import torchvision.utils as vutils
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime

from model import VAE, Encoder
from data import get_mnist_dataloaders
from loss import gaussian_vae_loss, beta_vae_loss
from latent_space import plot_latent_space, interpolate_latent_space
from decoder_inference import optimise_latent_vector, inpaint_right_half, decoder_mean_image


class Trainer:
    def __init__(
        self,
        output_dist="gaussian",
        latent_dim=20,
        epochs=50,
        batch_size=128,
        patience=5,
        lr=1e-3,
        device=None,
        overfit=False,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("experiments", output_dist, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_dist = output_dist
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.overfit = overfit

        self.model = VAE(latent_dim=self.latent_dim, output_dist=self.output_dist).to(
            self.device
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.overfit:
            full_loader, _, _ = get_mnist_dataloaders(batch_size=self.batch_size)
            single_batch = next(iter(full_loader))
            self.train_loader = [(single_batch[0], single_batch[1])]
            self.val_loader = self.train_loader
            self.test_loader = self.train_loader
        else:
            self.train_loader, self.val_loader, self.test_loader = (
                get_mnist_dataloaders(batch_size=self.batch_size)
            )

        # save config
        config = {
            "output_dist": self.output_dist,
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "lr": self.lr,
            "overfit": self.overfit,
        }
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def loss_fn(self, x, recon_out, mu, logvar):
        if self.output_dist == "gaussian":
            return gaussian_vae_loss(x, *recon_out, mu, logvar)
        elif self.output_dist == "beta":
            return beta_vae_loss(x, *recon_out, mu, logvar)
        else:
            raise ValueError("output_dist must be 'gaussian' or 'beta'")

    def train(self):
        best_val_elbo = float("-inf")
        best_epoch = 0
        train_elbos, val_elbos = [], []

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for x, _ in self.train_loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()

                recon_out, mu, logvar, _ = self.model(x)
                loss = self.loss_fn(x, recon_out, mu, logvar)

                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            train_elbo = -total_train_loss / len(self.train_loader)
            train_elbos.append(train_elbo)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x, _ in self.val_loader:
                    x = x.to(self.device)
                    recon_out, mu, logvar, _ = self.model(x)
                    loss = self.loss_fn(x, recon_out, mu, logvar)
                    total_val_loss += loss.item()

            val_elbo = -total_val_loss / len(self.val_loader)
            val_elbos.append(val_elbo)

            print(
                f"Epoch {epoch+1}: Train ELBO = {train_elbo}, Val ELBO = {val_elbo}"
            )

            if val_elbo > best_val_elbo:
                best_val_elbo = val_elbo
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "best_model.pt"),
                )
            elif epoch - best_epoch >= self.patience:
                print("Early stopping triggered.")
                break

        # Plot ELBO
        plt.plot(train_elbos, label="Train ELBO")
        plt.plot(val_elbos, label="Validation ELBO")
        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
        plt.title(f"Training Curve ({self.output_dist.capitalize()} VAE)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "elbo_curve.png"))
        plt.close()

        print(
            f"Training complete. Best model at epoch {best_epoch+1} (Val ELBO = {best_val_elbo})"
        )

        self.model.load_state_dict(
            torch.load(os.path.join(self.output_dir, "best_model.pt"))
        )
        self.evaluate_model_on_test()

    @torch.no_grad()
    def evaluate_model_on_test(self):
        self.model.eval()
        num_images = 32

        # Reconstruction
        test_batch = next(iter(self.test_loader))[0][:num_images].to(self.device)
        recon_out, mu, logvar, _ = self.model(test_batch)

        if self.output_dist == "gaussian":
            recon_mu, recon_logvar = recon_out
            eps = torch.randn_like(recon_mu)
            recon = recon_mu + torch.exp(0.5 * recon_logvar) * eps
        else:
            rec_alpha, rec_beta = recon_out
            recon = rec_alpha / (rec_alpha + rec_beta)

        recon = recon.clamp(0, 1)
        comparison = torch.cat([test_batch, recon], dim=0)
        grid = vutils.make_grid(comparison, nrow=num_images, pad_value=1)
        vutils.save_image(
            grid, os.path.join(self.output_dir, "test_reconstructions.png")
        )

        # Sampling from prior
        z_dim = mu.shape[1]
        z = torch.randn(num_images, z_dim).to(self.device)
        decoder_out = self.model.decoder(z)

        if self.output_dist == "gaussian":
            gen_mu, gen_logvar = decoder_out
            eps = torch.randn_like(gen_mu)
            samples = gen_mu + torch.exp(0.5 * gen_logvar) * eps
        else:
            alpha, beta = decoder_out
            samples = alpha / (alpha + beta)

        samples = samples.clamp(0, 1)
        grid = vutils.make_grid(samples.cpu(), nrow=8, pad_value=1)
        vutils.save_image(grid, os.path.join(self.output_dir, "samples_from_prior.png"))
        
        
        
    def visualize_latent_space(self,  num_samples: int = 1000, use_pca: bool = False):
        
        if self.latent_dim > 2:
            self.model.load_state_dict(
                torch.load(os.path.join(self.output_dir, r"C:\Users\Tommaso\Desktop\UNIVERSITA\ARTIFICIAL INTELLIGENCE\Generative AI\VAE_third_assignment\SAMU20-A3\experiments\gaussian\2025-06-15_07-40-51\best_model.pt"), map_location=torch.device('cpu'))
            )
            use_pca = True
        else:
            self.model.load_state_dict(
                torch.load(os.path.join(self.output_dir, r"path to trained model with 2 laten space dim"), map_location=torch.device('cpu'))
            )
        self.model.eval()
        '''
        test_x, test_labels = next(iter(self.test_loader))
        test_x = test_x[:num_samples]
        test_labels = test_labels[:num_samples]

        # (batch, 1, 28, 28) -> (batch, 28, 28) -> (batch, 1, 28, 28)
        test_input = test_x.reshape(test_x.shape[0], 28, 28).unsqueeze(1)
        '''
        dataset = self.test_loader.dataset  # accede direttamente al dataset completo
        test_x = torch.stack([dataset[i][0] for i in range(num_samples)])  # shape [1000, 1, 28, 28]
        test_labels = torch.tensor([dataset[i][1] for i in range(num_samples)])  # shape [1000]
        plot_latent_space(self.model.encoder, test_x, test_labels, use_pca=use_pca)
        
        
        
    def visualize_interpolation(self):
        test_x, test_labels = next(iter(self.test_loader))
        
        if self.latent_dim > 2:
            self.model.load_state_dict(
                torch.load(os.path.join(self.output_dir, r"C:\Users\Tommaso\Desktop\UNIVERSITA\ARTIFICIAL INTELLIGENCE\Generative AI\VAE_third_assignment\SAMU20-A3\experiments\gaussian\2025-06-15_07-40-51\best_model.pt"), map_location=torch.device('cpu'))
            )
        else:
            self.model.load_state_dict(
                torch.load(os.path.join(self.output_dir, r"path to trained model with 2 laten space dim"), map_location=torch.device('cpu'))
            )
        self.model.eval()
        test_x, test_labels = next(iter(self.test_loader))
    
        interpolate_latent_space(
            self.model,
            test_x,
            test_labels,
            device=self.device,
            sample=False     
        )
    # Task 4 utilities
    def task4a_three_column_grid(self, images_to_do: int = 4):
        """
        Creates the 3-column figure for 4a
        with the original | decoder-only reconstruction  | encoder reconstruction
        and saves it as task4a_recon_grid.png
        """
        self.model.load_state_dict(
                torch.load(os.path.join(self.output_dir, r"C:\Users\20212830\OneDrive - TU Eindhoven\Documents\TUe\2AMU20 (Generative AI Models)\SAMU20-A3\experiments\beta\BEST\best_model.pt"), map_location=torch.device('cpu'))
            )
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            self.test_loader.dataset, batch_size=1, shuffle=True
        )

        rows = []
        for i, (single_image, _) in enumerate(data_loader):
            if i >= images_to_do:
                break

            single_image = single_image.to(self.device)

            decoder_only_recon, _ = optimise_latent_vector(
                self.model, single_image
            )

            with torch.no_grad():
                mu, logvar = self.model.encoder(single_image)
                latent_code = mu
                decoder_output = self.model.decoder(latent_code)
                encoder_recon = decoder_mean_image(
                    decoder_output, self.model.output_dist
                )

            row = torch.cat(
                [single_image.cpu(),
                decoder_only_recon.cpu().unsqueeze(0),
                encoder_recon.cpu()], dim=0
            )
            rows.append(row)

        grid = torch.cat(rows, dim=0)

        save_path = os.path.join(self.output_dir, "task4a_recon_grid.png")
        torchvision.utils.save_image(grid, save_path,
                                    nrow=3,           # three columns per row
                                    normalize=True)
        print(f"reconstruction grid saved to {save_path}")


    def task4_inpaint_right_half(self, images_to_do=4):
        """
        Mask the right half of a few images, optimise z on the visible
        left half, then let the decoder complete the missing pixels.
        Output grid: original | masked input | completion for each row.
        """
        self.model.load_state_dict(
                torch.load(os.path.join(self.output_dir, r"C:\Users\20212830\OneDrive - TU Eindhoven\Documents\TUe\2AMU20 (Generative AI Models)\SAMU20-A3\experiments\beta\BEST\best_model.pt"), map_location=torch.device('cpu'))
            )
        self.model.eval()
        loader = torch.utils.data.DataLoader(
            self.test_loader.dataset, batch_size=1, shuffle=True
        )

        rows = []
        for i, (single_image, _) in enumerate(loader):
            if i >= images_to_do:
                break
            single_image = single_image.to(self.device)

            # make a copy whose right half is zero
            masked = single_image.clone()
            masked[..., :, masked.shape[-1] // 2 :] = 0

            completed, _ = inpaint_right_half(self.model, masked)

            rows.append(torch.cat(
                [single_image.cpu(),
                masked.cpu(),
                completed.cpu().unsqueeze(0)], dim=0))

        grid = torch.cat(rows, dim=0)
        save_path = os.path.join(self.output_dir, "task4_inpaint_right_half.png")
        torchvision.utils.save_image(grid, save_path,
                                    nrow=3, normalize=True)
        print(f"image saved to {save_path}")