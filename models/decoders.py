from torch import nn

class linear_decoder(nn.Module): # Identity function to decode from embedding
    def __init__(self, input_dim): # Specify input dimensions e.g. 2 for a 2D embedding
        super(linear_decoder, self).__init__()
        self.decoder = nn.Linear(in_features=input_dim, out_features=1)
 
    def forward(self, x):
        prediction = self.decoder(x)
        return prediction