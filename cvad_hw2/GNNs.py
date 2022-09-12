import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils


class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        input_size = hidden_size * 2
        # initialize self attention weights
        self.W_Q = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.W_K = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.W_V = torch.nn.Linear(input_size, hidden_size, bias=False) 
        
    def forward(self, hidden_states, attention_mask=None, mapping=None):
        """
        hidden_states.shape = (batch_size, max(#polylines), hidden_size*2)
        """
        # Project to Key, Query, Value
        # q_agent_features = hidden_states[:,0,:] .unsqueeze(1) # --> [batch_size, 1, hidden_size*2] features of the agent
        P_q = self.W_Q(hidden_states) # --> (batch_size, max(#polylines), hidden_size) i.e (batch_size, #Queries, hidden_size)
        P_k = self.W_K(hidden_states) # --> (batch_size, max(#polylines), hidden_size) i.e (batch_size, #Keys, hidden_size)
        P_v = self.W_V(hidden_states) # --> (batch_size, max(#polylines), hidden_size) i.e (batch_size, #Values, hidden_size)

        # Calculate the Attention Scores (i.e. Q * K.transpose())
        out = torch.bmm(P_q, P_k.transpose(1, 2)) # --> (batch_size, max(#polylines), max(#polylines)) i.e (batch_size, #Queries, #Keys)
        # mask Scores to prevent backprop for placeholder polylines which do not exist at a given scene(batch) but does have a placeholder due to dim = max(#polylines)
        if attention_mask is not None:
            out = out.masked_fill(attention_mask == 0, -1e9)
        # Calculate final Weighted Scores (i.e. Attention_Scores(out) * V)
        out = torch.bmm(torch.nn.functional.softmax(out, dim=-1), P_v) # --> (batch_size, max(#polylines), hidden_size) i.e (batch_size, #Queries, hidden_size)
        # Note that 0^th index(1^st row) of out Matrix for every batch has the values for the agent of interest (which had Query of 0^th index (1^st row) in the Query Matrix)
        return out



class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()

        input_size = hidden_size
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU()
        )

    def forward(self, hidden_states, lengths):
        """
        Note that it does not process in batches (batches = multiple scenes). Instead it takes in polylines of a certain scene at once.
        Inputs:
            hidden_states (torch.Tensor): Full tensor of polylines
                                hidden_states.shape = (#polylines, max(len(polylines)), hidden_size)
                                -> hidden_states -> (3, 19, 64) 
            lengths (list of int): A list of integers where each items correponds to
                                   number of vectors in that polyline
        Returns:
            poly_features.shape = (#polylines, hidden_size*2)
        """
        hidden_size = hidden_states.shape[2]
        poly_features = torch.zeros(hidden_states.shape[0], hidden_size * 2) # placeholder for the final extracted polyline features --> [#polylines, hidden_size * 2]
        
        for i, (cur_polyline, cur_length) in enumerate(zip(hidden_states, lengths)):
            cur_hidden_states = cur_polyline[:cur_length] # reduce max(len(polylines)) to current polylines feature vectors only
            ### l_0
            # g_enc (MLP + Batch_Norm + ReLU)
            x = self.layer1(cur_hidden_states) # --> [#polyline_vectors, hidden_size]
            # phi_agg (max pooling)
            phi_agg, _ = torch.max(x, dim=0) # --> [hidden_size] # Note that we max pool the feature vectors coordinate wise
            phi_agg = phi_agg.unsqueeze(0) # --> [1, hidden_size] # Note that this equals the aggregated vector which will be concatenated to the other vectors
            phi_agg = phi_agg.repeat(cur_length, 1) # --> [#polyline_vectors, hidden_size]
            # phi_rel
            out = torch.cat([x, phi_agg], dim=1) # --> [#polyline_vectors, (hidden_size * 2)]
            
            ### l_1
            # g_enc (MLP + Batch_Norm + ReLU)
            x = self.layer2(out) # --> [#polyline_vectors, hidden_size]
            # phi_agg (max pooling)
            phi_agg, _ = torch.max(x, dim=0) # --> [hidden_size]
            phi_agg = phi_agg.unsqueeze(0) # --> [1, hidden_size]
            phi_agg = phi_agg.repeat(cur_length, 1) # --> [#polyline_vectors, hidden_size]
            # phi_rel
            out = torch.cat([x, phi_agg], dim=1) # --> [#polyline_vectors, (hidden_size * 2)]
            
            ### l_2
            # g_enc (MLP + Batch_Norm + ReLU)
            x = self.layer3(out) # --> [#polyline_vectors, hidden_size]
            # phi_agg (max pooling)
            phi_agg, _ = torch.max(x, dim=0) # --> [hidden_size]
            phi_agg = phi_agg.unsqueeze(0) # --> [1, hidden_size]
            phi_agg = phi_agg.repeat(cur_length, 1) # --> [#polyline_vectors, hidden_size]
            # phi_rel
            out = torch.cat([x, phi_agg], dim=1) # --> [#polyline_vectors, (hidden_size * 2)]
            
            # Finally, obtain Polyline level features (i.e. reduce down to 1 feature vector to summarize each Polyline in the scene)
            cur_poly_features, _ = torch.max(out, dim=0)  # --> [(hidden_size * 2)]
            
            # Record the extracted polyline vector for the current polyline
            poly_features[i] = cur_poly_features

        return poly_features


class MLP(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        out = self.model(input)
        return out