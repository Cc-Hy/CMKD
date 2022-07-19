import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# a simple conv layer to match dimension
class CONV_DOWN(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()

        self.model_cfg = model_cfg

        self.num_bev_features = input_channels*2

        self.down_layer = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, stride=2, padding=1)

    def forward(self, data_dict):
        """ 
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        x = spatial_features

        x = self.down_layer(x)
        
        data_dict['spatial_features'] = x

        return data_dict
