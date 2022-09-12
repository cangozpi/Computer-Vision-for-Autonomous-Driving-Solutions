import numpy as np
class RewardManager():
    """Computes and returns rewards based on states and actions."""
    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        reward_dict = {}
        # Your code here
        # Rewards used:
        # diversion from optimal speed, diversion from optimal lane angle to prevent oscillations
        # in the middle of the lane, collisions are punished, distance to waypoint to make sure
        # the agents heads towards the desired location, distance to hazard if one is present to make
        # sure that agent doesn't crash into hazards, distance to traffic light if any red/yellow light
        # is on to make sure car obeys traffic rules.
     #   reward_dict["speed"] =   -(state["optimal_speed"] - state["speed"]) ** 2 
     #   reward_dict["angle"] =   -(state["waypoint_angle"] - state["lane_angle"]) ** 2
     #   reward_dict["collision"] = -1000 if state["collision"] == True else 0
     #   reward_dict["dist"] = - state["waypoint_dist"]
     #   reward_dict["hazard"] = -(state["hazard_dist"]) ** 2 if state["hazard"] else 0
     #   reward_dict["tl"] = -state["tl_dist"] if state["tl_state"] == 1 else 0

        # Reward implementation of the paper mentioned in the homework pdf
        # reward_dict["speed_reward"] =   min(abs(state["optimal_speed"]/(state["speed"]+1e-13)), abs(state["speed"]/(state["optimal_speed"]+1e-13))) # 1 if speed is optimal else linearly goes down to 0
        # d_max = 10 # assume maximum distance from the lane as 2 meters
        # reward_dict["position_reward"] = (-1 / d_max) * abs(state["lane_dist"]) 
        # # Note that without the roratation_reward, the agent would oscillate around the center lane.
        # reward_dict["rotation_reward"] = -1 * abs(state["route_angle"]) # prevents oscillations around the lane center

        reward_dict['collision'] = state['collision'] * -10
        reward_dict['speed'] = 1 - np.abs(state['optimal_speed'] - state['speed']) / 100

        if state['command'] == 3: # lane follow
            reward_dict['steer'] = -np.abs(state['lane_angle'] - action['steer'])

        # if state['command'] == 2: # staright
        #     reward_dict['steer'] = 1 - np.abs(action['steer'])

        # stopping if optimal speed is less than the speed of the car
        if state['optimal_speed'] < state['speed']:
            # time to stop linearly reagrding of the action
            diff = state['speed'] - state['optimal_speed']
            reward_dict['stopping'] = 1 - np.abs(action['brake'] - diff/100)

        # acceleration
        if state['optimal_speed'] > state['speed']:
            # time to stop linearly reagrding of the action
            diff = state['optimal_speed'] - state['speed']
            reward_dict['speeding'] = 1 - np.abs(action['throttle'] - diff/100)

        # stear (angle should be aligned with waypoint)
        reward_dict['way_steer'] = -np.abs(state['waypoint_angle'] - action['steer'])

        
        
        # Your code here
        reward = 0.0
        for val in reward_dict.values():
            reward += val
        reward_dict["reward"] = reward
        return reward_dict
