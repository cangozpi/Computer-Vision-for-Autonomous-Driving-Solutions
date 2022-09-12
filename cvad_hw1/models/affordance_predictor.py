import torch.nn as nn
# imports below are added by me
import torch
import torchvision

class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Feature Extractor
        vgg16 = torchvision.models.vgg16(pretrained=True).eval()
        self.feature_extractor_model = torch.nn.Sequential(*list(vgg16.children())[:-1]) # Remove classification head / Note that list(vgg16.children())[:-1] also equals vgg16.features()
        self.feature_extractor_model = self.feature_extractor_model.eval()
        # do not train the feature extractor model
        for param in self.feature_extractor_model.parameters():
            param.requires_grad = False

        feaure_extractor_out_dim = 25088 # Flattened dim of feature_extractor_model output

        # Initialize Task blocks 
        self.lane_dist_model = create_conditioned_affordance_regression_layer(feaure_extractor_out_dim) # conditioned on command
        self.route_angle_model = create_conditioned_affordance_regression_layer(feaure_extractor_out_dim) # conditioned on command
        self.tl_dist_model = create_not_conditioned_affordance_regression_layer(feaure_extractor_out_dim) # not conditioned on command
        self.tl_state_model = create_not_conditioned_affordance_classification_layer(feaure_extractor_out_dim) # not conditioned on command
        
        


        
    def forward(self, img, command):
        # Extract the Features
        features = self.feature_extractor_model(img)
        
        # Flatten features for Task blocks (fc layer requires flattening conv2d outputs)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)

        # Pass through Task Blocks:
        # predict lane distance (Regression)
        pred_lane_dist = self.lane_dist_model(features) # --> [N, 4]
        # condition the outputs on the command
        batch_size = pred_lane_dist.shape[0]
        pred_lane_dist = pred_lane_dist[torch.arange(batch_size), command] # --> [N]
        #pred_lane_dist = torch.squeeze(pred_lane_dist, dim=1) # --> [N]

        # predict route angle (Regression)
        pred_route_angle = self.route_angle_model(features) # --> [N, 4]
        pred_route_angle = pred_route_angle[torch.arange(batch_size), command] # --> [N]
        #pred_route_angle = torch.squeeze(pred_route_angle, dim=1) # --> [N]

        # predict traffic light distance (Regression)
        pred_tl_dist = self.tl_dist_model(features) # --> [N, 1]
        pred_tl_dist = torch.squeeze(pred_tl_dist, dim=1) # --> [N]

        # predict traffic light state (Classification)
        pred_tl_state = self.tl_state_model(features) # --> [N, 2]

        return pred_lane_dist, pred_route_angle, pred_tl_dist, pred_tl_state



def create_conditioned_affordance_regression_layer(in_dim):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 4) # output dim = 4*1 since there are 4 possible commands to condition on and regression is 1D
    )
    return model

def create_not_conditioned_affordance_regression_layer(in_dim):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1)
    )
    return model

def create_not_conditioned_affordance_classification_layer(in_dim):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 2)
    )
    return model