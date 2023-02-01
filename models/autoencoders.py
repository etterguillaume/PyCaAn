from torch import nn

class AE_MLP(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, latent_dim, output_dim):
        super(AE_MLP, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=output_dim),
            nn.BatchNorm1d(output_dim),
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