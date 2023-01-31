from torch import nn
import torch
import torch.nn.functional as F

class AE_MLP(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, latent_dim, output_dim, DO_rate=0):
        super(AE_MLP, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Dropout(p=DO_rate) # Portion of units that will be corrupted. 0 by default
            nn.Linear(in_features=input_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_channels=latent_dim, out_channels=output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        
        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=input_dim),
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

class conv_VAE(nn.Module):
    def __init__(self, h_dim, latent_dim):
        super(conv_VAE, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.fc1_mu = nn.Conv2d(in_channels=h_dim, out_channels=latent_dim,kernel_size=(1,1))
        self.fc1_log_var = nn.Conv2d(in_channels=h_dim, out_channels=latent_dim,kernel_size=(1,1))
        self.fc2 = nn.Conv2d(in_channels=latent_dim, out_channels=h_dim,kernel_size=(1,1))
        
        # decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=7, stride=7, padding=0),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
 
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc1_mu(x)
        log_var = self.fc1_log_var(x)
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var, x

def vae_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD