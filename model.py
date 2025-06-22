import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)   # (batch, 32, 14, 14)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # (batch, 64, 7, 7)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1) # (batch, 128, 7, 7)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dist='gaussian'):
        super(Decoder, self).__init__()
        self.output_dist = output_dist.lower()

        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 1, 1)  # (batch, 64, 7, 7)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # (batch, 32, 14, 14)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)   # (batch, 16, 28, 28)
        self.bn3 = nn.BatchNorm2d(16)
        self.final = nn.Conv2d(16, 1, 3, 1, 1)                # (batch, 1, 28, 28)

        if self.output_dist == 'gaussian':
            self.out_mu = nn.Conv2d(1, 1, 1)
            self.out_logvar = nn.Conv2d(1, 1, 1)
        elif self.output_dist == 'beta':
            self.out_alpha = nn.Conv2d(1, 1, 1)
            self.out_beta = nn.Conv2d(1, 1, 1)
        else:
            raise ValueError("output_dist must be 'gaussian' or 'beta'")

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.final(x)

        if self.output_dist == 'gaussian':
            mu = self.out_mu(x)
            logvar = self.out_logvar(x)
            logvar = torch.clamp(logvar, min=-6.0, max=2.0)
            return mu, logvar

        elif self.output_dist == 'beta':
            eps = 1
            alpha = F.softplus(self.out_alpha(x)) + eps
            beta = F.softplus(self.out_beta(x)) + eps
            return alpha, beta

class VAE(nn.Module):
    def __init__(self, latent_dim, output_dist='gaussian'):
        super(VAE, self).__init__()
        self.output_dist = output_dist.lower()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, output_dist)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoder_output = self.decoder(z)
        return decoder_output, mu, logvar, z
