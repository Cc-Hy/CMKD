import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

class SCConv(nn.Module):
    def __init__(self, input_channels, output_channels, pooling_r, norm_layer=nn.ReLU):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                                padding=1),
                    norm_layer(output_channels),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                                padding=1),
                    norm_layer(output_channels),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1,
                                padding=1),
                    norm_layer(output_channels),
                    )

    @autocast(enabled=False)
    def forward(self, x):
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])) + 1e-8) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """

    def __init__(self, input_channels, output_channels, pooling_r=4, norm_layer=nn.BatchNorm2d):
        super(SCBottleneck, self).__init__()
        group_width = int(input_channels /2)
        self.conv1_a = nn.Conv2d(input_channels, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(input_channels, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, padding=1, stride=1),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, pooling_r=pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, output_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(output_channels)

        self.relu = nn.ReLU(inplace=True)

    @autocast(enabled=False)
    def forward(self, x):

        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class SCNET(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()

        self.model_cfg = model_cfg

        self.num_bev_features = input_channels

        self.layers = self.make_layer(SCBottleneck, input_channels, num_blocks = model_cfg.NUM_BLOCKS)

    @autocast(enabled=False)
    def forward(self, data_dict):
        """ 
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        x = spatial_features.float()

        x = self.layers(x)
        
        data_dict['spatial_features'] = x

        return data_dict
    
    def make_layer(self, block, input_channels, num_blocks, norm_layer = nn.BatchNorm2d):

        layers = []

        for i in range(1, num_blocks):
            layers.append(block(input_channels, input_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

class SCNET_DOWN(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.get('NUM_OUT_FEATURES', input_channels*2)

        self.down_layer = nn.Sequential(
        nn.Conv2d(input_channels, self.num_bev_features, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
        )

        self.layers = self.make_layer(SCBottleneck, self.num_bev_features, num_blocks = model_cfg.NUM_BLOCKS)

    @autocast(enabled=False)
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

        x = self.layers(x)
        
        data_dict['spatial_features'] = x

        return data_dict
    
    def make_layer(self, block, input_channels, num_blocks, norm_layer = nn.BatchNorm2d):

        layers = []

        for i in range(1, num_blocks):
            layers.append(block(input_channels, input_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)