import torch.nn as nn
# imports below are added by me
import torch

class MultiLayerPolicy(nn.Module):
    """An MLP based policy network"""
    def __init__(self):
        super().__init__()
        self.input_dim = 11 # For using state dict directly. Should match the output dim of TD3._extract_features()
        #self.input_dim = 4  # For using Predicted Affordances. Uncomment this line and comment above for switching to predicted affordances
        self.output_dim = 2
        layers = [torch.nn.Linear(self.input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,4),
            torch.nn.ReLU(),
            torch.nn.Linear(4,self.output_dim*4), # Note that *4 since there are 4 possible commands
            torch.nn.Tanh() # tanh is final layer since both actions must be in the range [-1,1]
            ]
        self.model = torch.nn.Sequential(*layers)
        # layers = [torch.nn.Linear(self.input_dim, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32,64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64,32),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32,4),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(4,self.output_dim), 
        #     torch.nn.Tanh() # tanh is final layer since both actions must be in the range [-1,1]
        #     ]

        # # create separate models for each command to be able to condition on the given commands
        # self.model_c0 = torch.nn.Sequential(*layers)
        # self.model_c1 = torch.nn.Sequential(*layers)
        # self.model_c2 = torch.nn.Sequential(*layers)
        # self.model_c3 = torch.nn.Sequential(*layers)  
        

    def forward(self, features, command):
        """
        output should be a tensor of shape [B, 2], which will be used as an acceleration
        and steering value. Both actions must be in the range [-1, 1].
        """
        # condition on the command
        # batch_out = torch.tensor([])
        # for f, c in zip(features, command):
        #     f = f.unsqueeze(0) # convert [f] --> [1, f] for NN batch processing
        #     if c == 0: # LEFT
        #         out = self.model_c0(f) # [1, 2]
        #         batch_out = torch.cat((batch_out, out), dim=0)
        #     elif c == 1: # RIGHT
        #         out = self.model_c1(f) # [1, 2]
        #         batch_out = torch.cat((batch_out, out), dim=0)
        #     elif c == 2: # STRAIGHT
        #         out = self.model_c2(f) # [1, 2]
        #         batch_out = torch.cat((batch_out, out), dim=0)
        #     elif c == 3: # LANEFOLLOW
        #         out = self.model_c3(f) # [1, 2]
        #         batch_out = torch.cat((batch_out, out), dim=0)
        #     else:
        #         print("Unknown command passed in to policy.forward() !")
        # return batch_out
        # Below is an alternative approach to conditioning on command but q_loss and policy_loss functions should be modified accordingly
        out = self.model(features) # [Batch_size, 2*4], Note that * 4 since there are 4 possible commands
        # choose outputs which correspond to the given commands
        out_1 = out[torch.arange(out.size(0)), (command*2)] # [Batch_size, 1]
        out_2 = out[torch.arange(out.size(0)), (command*2)+1] # [Batch_size, 1]
        out = torch.cat((out_1.unsqueeze(1), out_2.unsqueeze(1)), dim=1) # [Batch_size, 2]
        
        return out
