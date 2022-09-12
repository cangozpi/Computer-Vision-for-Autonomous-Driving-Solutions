from torch.utils.data import Dataset
# imports below are added by me
import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root):
        self.data_root = data_root
        # Your code here
        # ${data_root}/measurements & ${data_root}/rgb holds data
        self.measurements_dir = os.path.join(data_root, "measurements")
        self.rgb_dir = os.path.join(data_root, "rgb")
        # read in file names
        self.file_names_measurements, self.file_names_rgb = read_file_names(self.measurements_dir, self.rgb_dir)
        
        # read in data
        self.measurements = []
        self.rgb = []
        # Record the data into arrays in corresponding order
        for file_name in self.file_names_measurements:
            # read in json measurement
            cur_measurement_dir = file_name + ".json"
            self.measurements.append(cur_measurement_dir)
            
        # for file_name in self.file_names_rgb:
        for file_name in self.file_names_rgb:
            # read in rgb image files
            cur_rgb_dir = file_name + ".png"
            self.rgb.append(cur_rgb_dir)

    def __getitem__(self, index):
        """
        Return RGB images and measurements.
        returns:
            rgb_images: torch.tensor of shape [H,W,C]
            measurement: dict
        """
        # Your code here
        cur_rgb_dir = self.rgb[index]
        resNet_input_size = (224,224) # ResNet expects input_size = (3,224,224)
        cur_rgb = Image.open(cur_rgb_dir).convert("RGB").resize(resNet_input_size) 
        cur_rgb_np = np.asarray(cur_rgb) # convert image to numpy ndarray
        rgb_image = torch.tensor(cur_rgb_np)
        rgb_image = rgb_image.permute(2,0,1) # convert [H,W,C] to [C,H,W]
        # preprocess the img for ResNet-18 backbone
        rgb_image = rgb_image.float()
        rgb_image /= 255. # normalize [0,255] to be in range [0,1]
        rgb_image = torchvision.transforms.functional.normalize(rgb_image, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        # Extract features
        cur_measurement_dir = self.measurements[index]
        f = open(cur_measurement_dir)
        measurement = json.loads(f.read())

        # Check for a mismatch that occured during img and measurement pairing during dataset initialization
        if cur_rgb_dir.split(".")[0].split("/")[-1] != cur_measurement_dir.split(".")[0].split("/")[-1]:
            print("Error. Wrong image and measurement matching detected in exper_dataset.py !")
        
        # extract measurements to form a tuple from the dict
        command = measurement["command"]
        speed = torch.tensor(measurement["speed"])
        expert_throttle = torch.tensor(measurement["throttle"])
        expert_brake = torch.tensor(measurement["brake"])
        expert_steer = torch.tensor(measurement["steer"])
        route_dist = measurement["route_dist"]
        route_angle = measurement["route_angle"]
        lane_dist = measurement["lane_dist"]
        lane_angle = measurement["lane_angle"]
        hazard = measurement["hazard"]
        hazard_dist = measurement["hazard_dist"]
        tl_state = measurement["tl_state"]
        tl_dist = measurement["tl_dist"]
        is_junction = measurement["is_junction"]
        measurements = [command, speed, expert_throttle, expert_brake, expert_steer, \
            route_dist, route_angle, lane_dist, lane_angle, hazard, hazard_dist, \
                tl_state, tl_dist, is_junction]
        f.close()
        return rgb_image, measurements 

    def __len__(self):
        return len(self.rgb)


# Auxilary helper functions for IO
def read_file_names(dir1, dir2):
    """
    Given directory returns the names of the files with the same name (except file extensions) that exist in both
    dir1 and dir2 directories. It returns them in the array format where same index returns corresponding file names. 
    """
    file_names1 = []
    file_names2 = []
    try:
        for filename in os.listdir(dir1):
            cur_dir1_name = os.path.join(dir1, filename)
            cur_dir2_name = os.path.join(dir2, filename)

            if os.path.isfile(cur_dir1_name) : # check if they are files (not folder)
                cur_dir1_name = cur_dir1_name.split(".")[0]
                cur_dir2_name = cur_dir2_name.split(".")[0]
                file_names1.append(cur_dir1_name)
                file_names2.append(cur_dir2_name)
    except:
        print("Error in read_file_names(), exper_dataset.py")

    return file_names1, file_names2