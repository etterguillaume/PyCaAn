import torch
from torch import nn

class AE_MLP(nn.Module): # Autoencoder with multilayer perceptron
    def __init__(self, input_dim, hidden_dims, output_dim): # TODO parameterize num_layer and dimensions as vector eg [64,32,16,8]
        super(AE_MLP, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dims[1], out_features=output_dim),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=input_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

class TCN(nn.Module): # Autoencoder with multilayer perceptron
    def __init__(self, input_dim, hidden_dims, output_dim, kernel_size): # TODO parameterize num_layer and dimensions as vector eg [64,32,16,8]
        super(TCN, self).__init__()
        self.block_length = kernel_size
        # encoder
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=kernel_size, stride=1,padding=0),
            nn.LeakyReLU()
            )
        self.encoder = nn.Sequential(
            nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dims[1], out_features=output_dim),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[0]),
            nn.LeakyReLU()
            )
        self.conv_out = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_dims[0], out_channels=input_dim, kernel_size=10, stride=1,padding=0)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_in(x)
        x = x.view(batch_size, -1) # Flatten
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        reconstruction = reconstruction.view(batch_size,-1,1) # Expand for transpose conv
        reconstruction = self.conv_out(reconstruction)
        return reconstruction, embedding

class bVAE(nn.Module): # Autoencoder with multilayer perceptron
    def __init__(self, input_dim, hidden_dims, output_dim): # TODO parameterize num_layer and dimensions as vector eg [64,32,16,8]
        super(bVAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=output_dim),
        )

        self.mu = nn.Linear(in_features=output_dim, out_features=output_dim)
        self.sigma = nn.Linear(in_features=output_dim, out_features=output_dim)
        #self.fc = nn.Linear(in_features=output_dim, out_features=output_dim)

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=input_dim),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # Encode
        embedding = self.encoder(x)

        # Sample
        mu = self.mu(embedding)
        sigma = self.sigma(embedding)
        z = self.reparameterize(mu, sigma)
        #z = self.fc(z)

        # Decode
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, mu, sigma

def bVAE_loss(bce_loss, mu, logvar, beta):
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
    return BCE + beta*KLD

class test_AE(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, hidden_dims, output_dim): # TODO parameterize num_layer and dimensions as vector eg [64,32,16,8]
        super(test_AE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim),
            nn.ReLU(),
            nn.Linear(in_features=input_dim, out_features=output_dim),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=input_dim),
            nn.ReLU(),
            nn.Linear(in_features=input_dim, out_features=input_dim),
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

class TCN_10(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TCN_10, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=2, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=output_dim, kernel_size=3, stride=1,padding=0),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=output_dim, out_channels=8, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=input_dim, kernel_size=2, stride=1,padding=0),
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding