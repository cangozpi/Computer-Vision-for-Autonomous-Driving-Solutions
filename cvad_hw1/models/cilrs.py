import torch.nn as nn
# imports below are added by me
import torch
import torchvision


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super().__init__()
        # image preprocessing required for resnet18
        # ResNet18 as backbone
        self.backbone_output_size = 512
        resNet = torchvision.models.resnet18(pretrained=True)#.eval()
        # we don't need classification layer (fc) so remove it
        backboneModules = list(resNet.children())[:-1]
        self.backbone_model = torch.nn.Sequential(*backboneModules) # output_size = [N, 512, 7, 7]
        self.backbone_model = self.backbone_model.eval()
        # do not train the backbone
        for param in self.backbone_model.parameters():
            param.requires_grad = False
        
        # Speed prediction layer
        self.speed_pred_model = torch.nn.Sequential(
            #torch.nn.Flatten(),
            torch.nn.Linear(self.backbone_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1) # 1D output for speed prediction
        )
        
        # Initialize model to be used as conditional module:
        self.conditional_module_model = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_output_size+1, 256), # Note that input dim = (backbone_output.dim + speed.dim(1D))
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 12) # Note that out_dim = 4*3 since (4 possible commands and 3 action predictions per each command)
        )


    def forward(self, img, command, speed):
        """
        given:
            img: RGB images of shape [N, C, H, W]
            commands: int in the range(4)
        returns:
            pred_speed: floatTensor of shape [N, 1] for predicted speed
            pred_actions: tuple of (throttle, brake, steer) as floatTensor in shape [N, 1] each 
        """ 
        # Extract Features through ResNet18 backbone
        features = self.backbone_model(img)
        # Flatten features (required for classification head (fc))
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)

        # predict speed using features
        pred_speed = self.speed_pred_model(features) # --> [N, 1]
        pred_speed = torch.squeeze(pred_speed, dim=1) # --> [N]
        

        # predict actions using features,speed and condition on command
        inp = torch.hstack((features, speed)) # input should be extracted features vector concatenated with speed vector
        actions = self.conditional_module_model(inp) # [Batch_size, 12]
        # Condition outputs on commands
        pred_throttle_actions = actions[torch.arange(batch_size), (3*command)] # [Batch_size]
        pred_brake_actions = actions[torch.arange(batch_size), (3*command)+1] # [Batch_size]
        pred_steer_actions = actions[torch.arange(batch_size), (3*command)+2] # [Batch_size]

                
        return (pred_speed, pred_throttle_actions, pred_brake_actions, pred_steer_actions)