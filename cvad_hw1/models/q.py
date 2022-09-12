import torch.nn as nn
# imports below are added by me
import torch

class MultiLayerQ(nn.Module):
    """Q network consisting of an MLP."""
    def __init__(self, config):
        super().__init__()
        self.input_dim = 11 + config["action_size"] # For using state dict directly. Should match the output dim of TD3._extract_features() + MultiLayerPolicy.forward()'s dim
        #self.input_dim = 4 + config["action_size"] # For using Predicted Affordances. Uncomment this line and comment above for switching to predicted affordances
        self.output_dim = 1 # a single Q(s,a) value
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 32), # self.input_dim = 13
            torch.nn.ReLU(),
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,4),
            torch.nn.ReLU(),
            torch.nn.Linear(4,self.output_dim))  
        

    def forward(self, features, actions):
        """
        You will be outputting a tensor of shape [B, 1] since you are outputting
         only a single q-value.
        """
        # Concatenate features and actions
        inp_vec = torch.cat((features, actions), dim=1)
        # Forward pass
        out = self.model(inp_vec)
        return out