import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate

from ...utils.loss_utils import QualityFocalLoss_no_reduction

class AnchorHeadSingleCMKD_V2(AnchorHeadTemplate):
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
        self.teacher_pred = None

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
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
            pass

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

        # student pred
        cls_preds = self.forward_ret_dict['cls_preds']
        batch_size = cls_preds.shape[0]
        cls_preds = cls_preds.view(batch_size, -1, self.num_class).contiguous()
        # teacher pred
        cls_teacher_pred = self.teacher_pred['cls_preds']
        cls_teacher_pred = cls_teacher_pred.view(batch_size, -1, self.num_class).contiguous()
        # to logits
        cls_teacher_pred = torch.sigmoid(cls_teacher_pred)

        # pos norm
        pos = (cls_teacher_pred > 0.1).float()
        pos_normalizer = pos.sum(1, keepdim=True).float()
        
        # calculate loss

        cls_loss = self.QFL(cls_preds, cls_teacher_pred, pos_normalizer)
        cls_loss = cls_loss.sum() 
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):

        # student pred
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        batch_size = box_preds.shape[0]

        # teacher pred
        box_teacher_pred = self.teacher_pred['box_preds']
        box_dir_cls_teacher_preds = self.teacher_pred.get('dir_cls_preds', None)

        # pos norm
        cls_teacher_pred = torch.sigmoid(self.teacher_pred['cls_preds'])
        # B,H,W,N_cls -> B,H*W
        positives = cls_teacher_pred.view(batch_size, -1, self.num_class).contiguous().sum(-1) > 0.1
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        reg_weights *= cls_teacher_pred.view(batch_size, -1, self.num_class).contiguous().sum(-1)

        box_preds = box_preds.view(batch_size, -1,
                            box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                            box_preds.shape[-1])

        box_teacher_pred = box_teacher_pred.view(batch_size, -1,
                            box_teacher_pred.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                            box_teacher_pred.shape[-1])

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_teacher_pred)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            weights *= cls_teacher_pred.view(batch_size, -1, self.num_class).contiguous().sum(-1)
            dir_targets = torch.sigmoid(box_dir_cls_teacher_preds).view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
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
