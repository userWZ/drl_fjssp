import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GraphNet(nn.Module):
    def __init__(self, num_node_features, pooling=False): # num_out_features
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        self.pooling = pooling
    
    