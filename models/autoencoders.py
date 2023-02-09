from torch import nn

class AE_MLP(nn.Module): # Autoencoder with multilayer perceptron backend and dropout input layer
    def __init__(self, input_dim, hidden_dims, output_dim): # TODO parameterize num_layer and dimensions as vector eg [64,32,16,8]
        super(AE_MLP, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(hidden_dims[0], out_features=hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[2], out_features=hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[3], out_features=output_dim),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[3], out_features=hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[2], out_features=hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=hidden_dims[0], out_features=input_dim),
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Conv1d(in_channels=8, out_channels=output_dim, kernel_size=3, stride=1,padding=0),
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=output_dim, out_channels=8, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.ConvTranspose1d(in_channels=64, out_channels=input_dim, kernel_size=2, stride=1,padding=0),
        )
 
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding