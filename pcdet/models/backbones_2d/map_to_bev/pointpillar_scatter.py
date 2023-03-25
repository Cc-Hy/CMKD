import torch
import torch.nn as nn
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        self.use_conv_layers = model_cfg.get('USE_CONV_LAYERS', False)
        if self.use_conv_layers:
            self.conv_layers = nn.ModuleList()
            self.conv_layer_num = model_cfg.get('CONV_LAYER_NUM', 5)
            self.use_ds_layer = model_cfg.get('USE_DS_LAYER', False)
            if self.use_ds_layer:
                ds_layer = BasicBlock2D(
                        in_channels=self.num_bev_features,
                        out_channels=self.num_bev_features,
                        kernel_size=3,padding=1,stride=2)
                self.conv_layers.append(ds_layer)
            for i in range(self.conv_layer_num):
                single_conv_layer = BasicBlock2D(
                        in_channels=self.num_bev_features,
                        out_channels=self.num_bev_features,
                        kernel_size=3,padding=1)
                self.conv_layers.append(single_conv_layer)

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)

        if self.use_conv_layers:
            for l in self.conv_layers:
                batch_spatial_features = l(batch_spatial_features)

        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
