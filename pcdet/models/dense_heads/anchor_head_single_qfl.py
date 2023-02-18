import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.loss_utils import QualityFocalLoss, QualityFocalLoss_no_reduction

class AnchorHeadSingleQFL(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        self.QFL = QualityFocalLoss_no_reduction(beta = 2.0)
        
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_cls_layer_loss(self):

        cls_preds = self.forward_ret_dict['cls_preds']
        box_preds = self.forward_ret_dict['box_preds']
        dir_cls_preds = self.forward_ret_dict['dir_cls_preds']
        gt_boxes = self.forward_ret_dict['gt_boxes']
        batch_size = cls_preds.shape[0]

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(       # decode    batch_cls_preds B,H*W*6 int
                    batch_size=batch_size,
                    cls_preds=cls_preds.detach(), box_preds=box_preds.detach(), dir_cls_preds=dir_cls_preds.detach()
                )        

        iou_targets = None
        cls_targets = None

        for i in range(batch_size):         
                iou_3d = iou3d_nms_utils.boxes_iou3d_gpu(batch_box_preds[i], gt_boxes[i][...,:7])       # M,N 
                iou_3d, indice = torch.max(iou_3d, dim=1)                                               # M,M
                
                iou_target = iou_3d.unsqueeze(dim=0)                                                    # 1,M                                                    
                if iou_targets == None:
                    iou_targets = iou_target
                else:
                    iou_targets = torch.cat((iou_targets,iou_target), dim=0)                            # B,M


                cls_target = gt_boxes[i][indice][...,-1].unsqueeze(0)
                if cls_targets == None:
                    cls_targets = cls_target                                                            # M ->1,M
                else:
                    cls_targets = torch.cat((cls_targets,cls_target), dim=0)                            # B,M
                
        pos = (iou_targets > 0).float()
        pos_normalizer = pos.sum(1, keepdim=True).float().unsqueeze(-1)

        one_hot_targets = torch.zeros(                  # B,M,C+1 (0,1,2,3) 0->bg
            *list(iou_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_preds.device
        )

        
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)             # B,M -> B,M,1, one-hot

        cls_targets = iou_targets.unsqueeze(-1) * one_hot_targets
        cls_targets = cls_targets[...,1:]

        cls_preds = cls_preds.view(batch_size,-1,cls_targets.shape[-1]).contiguous()        # B,-1,M

        cls_loss = self.QFL(cls_preds, cls_targets, pos_normalizer).sum()

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def forward(self, data_dict):

        batch_size = data_dict['batch_size']
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]  [B,H,W,C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:

            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)


        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        
        return data_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()

        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
    
        return rpn_loss, tb_dict
