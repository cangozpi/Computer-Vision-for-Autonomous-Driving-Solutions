import numpy as np
import torch
from func_timeout import FunctionTimedOut, func_timeout
from utils.rl_utils import generate_noisy_action_tensor

from .off_policy import BaseOffPolicy
# Uncomment below for switching to using predicted affordances in RL 
#from models.affordance_predictor import AffordancePredictor 

class TD3(BaseOffPolicy):

    # Uncomment below for switching to using Predicted Affordances
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.affordancePredictor = AffordancePredictor()


    def _compute_q_loss(self, data):
        """Compute q loss for given batch of data."""
        # Your code here
        stored_features, stored_command, stored_action, stored_reward, \
                stored_new_features, stored_new_command, stored_is_terminal = data # len(data) = 7
        # Calculate Q(s,a)
        q1_val = self.q_nets[0](stored_features, stored_action) # [Batch_size, 1]
        q2_val = self.q_nets[1](stored_features, stored_action) # [Batch_size, 1]
        q_val_estimates = (q1_val, q2_val) 

        # Calculate a_t+1 using target policy network
        with torch.no_grad(): # don't calculate gradients since target networks will be updated via polyak averaging
            target_action = self.target_policy(stored_new_features, stored_command.long()) # [Batch_size, 2]
            # Add noise to target action for smoothing purposes (i.e. target policy smoothing)
            target_action = generate_noisy_action_tensor(
                target_action, self.config["action_space"], self.config["policy_noise"], 1.0) # [Batch_size, 2]
            
            # Calculate TD Target
            q1_target_val = self.target_q_nets[0](stored_new_features, target_action)
            q2_target_val = self.target_q_nets[1](stored_new_features, target_action)
            # clip q values
            q_target_val = torch.min(q1_target_val, q2_target_val) # [Batch_size, 1]
            q_target_val = q_target_val.squeeze(1) # [Batch_size]
            td_target = stored_reward + self.config["discount"] * (1 - stored_is_terminal.long()) * q_target_val # [Batch_size]
            td_target = td_target.unsqueeze(1) # [Batch_size, 1]
        # calculate losses
        q1_loss = torch.square(q1_val - td_target).mean() # --> [1]
        q2_loss = torch.square(q2_val - td_target).mean() # --> [1]
        q_loss = (q1_loss + q2_loss) 
            
        return q_val_estimates, q_loss
      


    def _compute_p_loss(self, data):
        """Compute policy loss for given batch of data."""
        # Your code here
        p_loss = 0
        stored_features, stored_command, stored_action, stored_reward, \
                stored_new_features, stored_new_command, stored_is_terminal = data # len(data) = 7
        
        # Calculate a = policy(s)
        policy_action = self.policy(stored_features, stored_command.long()) # [Batch_size, 2]
        
        # Calculate Q(s,a=pi(s)) using 1'st Q Network
        q_val = self.q_nets[0](stored_features, policy_action) # [Batch_size, 1]
        
        # We want to maximize q_val so we'll minimize -q_val
        p_loss = -1 * q_val.sum()
        
        # calculate mean of loss
        batch_size = len(stored_reward)
        p_loss /= batch_size
        return p_loss
        
        

    def _extract_features(self, state):
        """Extract whatever features you wish to give as input to policy and q networks."""
        # Your code here
        # Uncomment below to switch using features from the state dict
        speed_val = torch.tensor(state["speed"], dtype=torch.float32) # float
        opt_speed = torch.tensor(state["optimal_speed"], dtype=torch.float32) # float
        wp_dist = torch.tensor(state["waypoint_dist"], dtype=torch.float32) # float
        command = torch.tensor(state["command"], dtype=torch.float32) # float
        lane_dist = torch.tensor(state["lane_dist"], dtype=torch.float32) # float
        lane_angle = torch.tensor(state["lane_angle"], dtype=torch.float32) # float
        is_junction = torch.tensor(int(state["is_junction"]), dtype=torch.float32) # bool
        hazard = torch.tensor(int(state["hazard"]), dtype=torch.float32) # bool
        hazard_dist = torch.tensor(state["hazard_dist"], dtype=torch.float32) # float
        tl_state = torch.tensor(state["tl_state"], dtype=torch.float32) # int
        tl_dist = torch.tensor(state["tl_dist"], dtype=torch.float32) # float
        features = (speed_val, opt_speed, wp_dist, command, lane_dist, lane_angle,\
            is_junction, hazard, hazard_dist, tl_state, tl_dist)


        # Uncomment below to use predicted affordances as the extracted features
        # pred_lane_dist, pred_route_angle, pred_tl_dist, pred_tl_state = self.affordancePredictor(state["rgb"], state["command"])
        # features = (pred_lane_dist, pred_route_angle, pred_tl_dist, pred_tl_state)

        return features
        

    def _take_step(self, state, action):
        try:
            action_dict = {
                "throttle": np.clip(action[0, 0].item(), 0, 1),
                "brake": abs(np.clip(action[0, 0].item(), -1, 0)),
                "steer": np.clip(action[0, 1].item(), -1, 1),
            }
            new_state, reward_dict, is_terminal = func_timeout(
                20, self.env.step, (action_dict,))
        except FunctionTimedOut:
            print("\nEnv.step did not return.")
            raise
        return new_state, reward_dict, is_terminal

    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        features = self._extract_features(state)
        features_input = torch.tensor([f.squeeze(0) for f in features]).unsqueeze(0) # Added by me !!!
        if self.step >= self.config["exploration_steps"]:
            action = self.policy(features_input, torch.tensor([state["command"]]).long())
            action = torch.unsqueeze(action, dim=0)
            action = generate_noisy_action_tensor(
                action, self.config["action_space"], self.config["policy_noise"], 1.0)
            action = action.squeeze(0)
            print("RL Agent throttle/brake: ", action[:,0].item(), ", steer: ", action[:,1].item())
        else:
            action = self._explorer.generate_action(state)
            #print(" Manual Controller (NOT RL Agent) throttle/brake: ", action)
        if self.step <= self.config["augment_steps"]:
            action = self._augmenter.augment_action(action, state)

        # Take step
        new_state, reward_dict, is_terminal = self._take_step(state, action)

        new_features = self._extract_features(new_state)

        # Prepare everything for storage
        stored_features = [f.detach().cpu().squeeze(0) for f in features]
        stored_command = state["command"]
        stored_action = action.detach().cpu().squeeze(0)
        stored_reward = torch.tensor([reward_dict["reward"]], dtype=torch.float)
        stored_new_features = [f.detach().cpu().squeeze(0) for f in new_features]
        stored_new_command = new_state["command"]
        stored_is_terminal = bool(is_terminal)
        
        self._replay_buffer.append(
            (stored_features, stored_command, stored_action, stored_reward,
             stored_new_features, stored_new_command, stored_is_terminal)
        )
        self._visualizer.visualize(new_state, stored_action, reward_dict)
        return reward_dict, new_state, is_terminal
