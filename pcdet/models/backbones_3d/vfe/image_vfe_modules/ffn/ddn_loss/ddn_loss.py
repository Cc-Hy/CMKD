import torch
import torch.nn as nn


from .balancer import Balancer
from pcdet.utils import transform_utils

try:
    from kornia.losses.focal import FocalLoss
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNLoss(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 disc_cfg,
                 fg_weight=None,
                 bg_weight=None,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        if fg_weight is not None and bg_weight is not None:
            self.balancer = Balancer(downsample_factor=downsample_factor,
                                    fg_weight=fg_weight,
                                    bg_weight=bg_weight)
        else:
            self.balancer = None

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d = None):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}

        # Bin depth map to create target
        depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)

        # Compute foreground/background balancing
        if self.balancer is not None and gt_boxes2d is not None:
            loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)
        else:
            loss = loss.mean()

        # Final loss
        loss *= self.weight
        tb_dict.update({"ddn_loss": loss.item()})

        return loss, tb_dict
