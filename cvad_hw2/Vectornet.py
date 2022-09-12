import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from GNNs import MLP, Global_Graph, Sub_Graph
from utils import (batch_init, get_dis_point_2_points, merge_tensors,
                   to_origin_coordinate)


class VectorNet(nn.Module):
    def __init__(self, hidden_size, device):
        super(VectorNet, self).__init__()

        self.sub_graph = Sub_Graph(hidden_size)
        self.global_graph = Global_Graph(hidden_size)
        self.predict_traj = MLP(hidden_size, 6*30*2 + 6)
        self.device = device
        self.hidden_size = hidden_size
        self.traj_completion_criterion = nn.SmoothL1Loss()

    def forward_encode_sub_graph(self, mapping, matrix, polyline_spans, batch_size):
        # len(input_list_list) = batch_size
        # len(input_list_list[i]) = #polylines in scene i
        # input_list_list[i][j] = #vectors in polyline j of scene i
        input_list_list = []
        for i in range(batch_size):
            input_list = []
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(
                    matrix[i][polyline_span], device=self.device)
                input_list.append(tensor)

            input_list_list.append(input_list)

        element_states_batch = []

        # Aim is getting a feature for each polyline
        #   from many vectors of it
        for batch_idx in range(batch_size):
            hidden_states, lengths = merge_tensors(
                input_list_list[batch_idx], self.device, self.hidden_size)
            # input_list_list[i] -> (8, 64), (6, 64), (19, 64)
            # hidden_states -> (3, 19, 64)

            hidden_states = self.sub_graph(hidden_states, lengths)
            # hidden_states.shape = (#polylines, hidden_size)
            element_states_batch.append(hidden_states)

        return element_states_batch

    def forward(self, mapping, validate=False):

        matrix = [i["matrix"] for i in mapping]
        polyline_spans = [i["polyline_spans"] for i in mapping]
        labels = [i["labels"] for i in mapping]

        batch_init(mapping)
        batch_size = len(matrix)

        element_states_batch = self.forward_encode_sub_graph(
            mapping, matrix, polyline_spans, batch_size)

        inputs, inputs_lengths = merge_tensors(
            element_states_batch, self.device, self.hidden_size*2) # !!! ı CHANGED self.hidden_size TO (self.hidden_size*2)
        # inputs.shape = (batch_size, max(#polylines), hidden_size*2)

        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros(
            [batch_size, max_poly_num, max_poly_num], device=self.device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask=attention_mask, mapping=mapping)

        outputs = self.predict_traj(hidden_states[:, 0, :])
        pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
        outputs = outputs[:, :- 6].view([batch_size, 6, 30, 2])

        ### YOUR CODE HERE ###
        for i in range(batch_size):
            gt_points = np.array(labels[i]).reshape([30, 2])
            # argmin = None  # find prediction with closest endpoint to gt
            # loss = None  # find loss over predicted trajectory argmin
            gt_points = torch.tensor(gt_points).to(self.device).float()
            preds = outputs[i] # --> [6, 30, 2] (i.e. 6 predicted trajectories for the current scene)
            closest_pred_index = 0 # index of the predicted trajectory with the closest endpoint to the gt
            closest_pred_l2_dist = 0 # l2 distance of the closest predicted trajectory to the gt
            for i, traj in enumerate(preds): # traj --> [30, 2] # iterate through each of the predicted trajectories
                cur_pred_x, cur_pred_y = traj[-1][0], traj[-1][1] 
                gt_x, gt_y = gt_points[-1][0], gt_points[-1][1]      

                cur_gt_l2_dist = ((cur_pred_x - gt_x)**2 + (cur_pred_y - gt_y)**2) ** (1/2)# L2 dist
                if (cur_gt_l2_dist <= closest_pred_l2_dist) or (i == 0):
                    closest_pred_index = i
                    closest_pred_l2_dist = cur_gt_l2_dist
            argmin = preds[closest_pred_index]  # prediction with closest endpoint to gt --> [30, 2]

            
            loss = torch.nn.functional.mse_loss(argmin, gt_points)  # find loss over closest predicted trajectory argmin and gt


        ### YOUR CODE HERE ###

        if validate:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(
            ), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    to_origin_coordinate(each, i)

            return outputs, pred_probs

        return loss
