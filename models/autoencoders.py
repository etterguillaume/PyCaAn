from torch import nn

class AE_MLP(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, latent_dim, output_dim):
        super(AE_MLP, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=latent_dim, out_features=output_dim),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=latent_dim, out_features=input_dim),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


class TCN_10(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(TCN_10, self).__init__()
 
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=latent_dim, kernel_size=2, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=latent_dim, out_channels=output_dim, kernel_size=3, stride=1,padding=0),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=output_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=latent_dim, out_channels=output_dim, kernel_size=2, stride=1,padding=0),
            nn.Sigmoid() # To scale output between [0,1]
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding