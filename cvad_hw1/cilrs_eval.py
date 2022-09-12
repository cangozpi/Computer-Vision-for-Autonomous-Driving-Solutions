import os

import yaml

from carla_env.env import Env
# imports below are added by me
from models.cilrs import CILRS
import torch
import torchvision
from PIL import Image
import numpy as np


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()

    def load_agent(self):
        # Your code here
        # initialize the model and load its weights
        save_path = "cilrs_model.ckpt"
        self.model = torch.load(save_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def generate_action(self, rgb, command, speed):
        # Your code here
        resNet_input_size = (224,224)
        cur_rgb = Image.fromarray(rgb.astype('uint8')).convert("RGB").resize(resNet_input_size)
        cur_rgb_np = np.asarray(cur_rgb) # convert image to numpy ndarray
        rgb_image = torch.tensor(cur_rgb_np)
        rgb_image = rgb_image.permute(2,0,1) # convert [H,W,C] to [C,H,W]
        # preprocess the img for ResNet-18 backbone
        rgb_image = rgb_image.float()
        rgb_image /= 255. # normalize [0,255] to be in range [0,1]
        rgb_image = torchvision.transforms.functional.normalize(rgb_image, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        rgb = rgb_image

        rgb = torch.unsqueeze(rgb, dim=0) # [1, C, H, W]

        if self.device != "cpu":
            rgb = rgb.cuda()
        _pred_speed, pred_throttle, pred_brake, pred_steer  = self.model(rgb.float(), command, speed)
        
        print("throttle: ", pred_throttle.item())
        print("brake: ", pred_brake.item())
        print("steer: ", pred_steer.item())
        # need to return (throttle, steer, brake)
        # TODO: not sure if the min(throttle, brake) should be set to 0 to allow acceleration
        if abs(pred_throttle) > abs(pred_brake):
            pred_brake *= 0
        else:
            pred_throttle *= 0

        return pred_throttle.item(), pred_steer.item(), pred_brake.item()

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        speed = torch.tensor([speed]).unsqueeze(1).float()
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
