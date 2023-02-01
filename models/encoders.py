from torch import nn

class encoder_MLP(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, latent_dim, output_dim):
        super(encoder_MLP, self).__init__()
 
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
        
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

class encoder_conv(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, latent_dim, output_dim, kernel_size, stride):
        super(encoder_MLP, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=latent_dim, kernel_size=kernel_size, stride=stride),
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
        
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding