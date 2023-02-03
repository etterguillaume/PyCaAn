from torch import nn

class position_decoder(nn.Module): # Identity function to decode from embedding
    def __init__(self, input_dims): # Specify input dimensions e.g. 2 for a 2D embedding
        super(position_decoder, self).__init__()
        self.decoder = nn.Linear(in_features=input_dims, out_features=2)
 
    def forward(self, x):
        prediction = self.decoder(x)
        return prediction