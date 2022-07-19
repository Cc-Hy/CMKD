import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from .target_assigner.axis_aligned_target_assigner_score import AxisAlignedTargetAssignerScore

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.loss_utils import QualityFocalLoss_no_reduction


# 1.take in gt_score, with AxisAlignedTargetAssignerScore
# 2.use QFL with IoU scores
class AnchorHeadSingleCMKD(AnchorHeadTemplate):
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

        self.QFL = QualityFocalLoss_no_reduction(beta=2.0)

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        if data_dict.get('gt_boxes', None) is not None:
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            gt_scores=data_dict['gt_scores']
            gt_scores[gt_scores == -1] = 1
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                gt_scores=gt_scores,
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

    def get_cls_layer_loss(self):

        cls_preds = self.forward_ret_dict['cls_preds']
        box_preds = self.forward_ret_dict['box_preds']
        dir_cls_preds = self.forward_ret_dict['dir_cls_preds']
        gt_boxes = self.forward_ret_dict['gt_boxes']
        batch_size = cls_preds.shape[0]

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
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

        one_hot_targets = torch.zeros(
            *list(iou_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_preds.device
        )

        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)             # B,M -> B,M,1, one-hot

        cls_targets = iou_targets.unsqueeze(-1) * one_hot_targets
        cls_targets = cls_targets[...,1:]

        cls_preds = cls_preds.view(batch_size,-1,cls_targets.shape[-1]).contiguous()

        cls_loss = self.QFL(cls_preds, cls_targets, pos_normalizer)
        scores = self.forward_ret_dict['scores'] * 2
        scores[scores==0]=1
        cls_loss = cls_loss * scores.unsqueeze(-1)
        cls_loss = cls_loss.sum()
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        
        scores = self.forward_ret_dict['scores'] * 2
        scores[scores==0]=1
        loc_loss_src = loc_loss_src * scores.unsqueeze(-1)

        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def assign_targets(self, gt_boxes, gt_scores):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes, gt_scores
        )
        return targets_dict

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssignerScore':
            target_assigner = AxisAlignedTargetAssignerScore(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner