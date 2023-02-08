from torch import nn

class linear_decoder(nn.Module): # Identity function to decode from embedding
    def __init__(self, input_dims, output_dims): # Specify input dimensions e.g. 2 for a 2D embedding
        super(linear_decoder, self).__init__()
        self.decoder = nn.Linear(in_features=input_dims, out_features=output_dims)
 
    def forward(self, x):
        prediction = self.decoder(x)
        return prediction

class TCN10_decoder(nn.Module): # Identity function to decode from embedding
    def __init__(self, in_channels, out_channels): # Specify input dimensions e.g. 2 for a 2D embedding
        super(TCN10_decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=10)
 
    def forward(self, x):
        prediction = self.decoder(x)
        return prediction