#
# @author:charlotte.Song
# @file: yolact_detectors.py
# @Date: 2019/9/16 19:55
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import mmcv

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, force_fp32, auto_fp16
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import torch.nn.functional as F
import pycocotools.mask as mask_util


@DETECTORS.register_module
class YolactDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,    # yolact_bbox_head.
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YolactDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg  = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(YolactDetector, self).init_weights(pretrained=pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        assert self.with_neck
        proto, x = self.neck(x)
        return proto, x

    # add function to crop instance mask.
    # extract_proto
    def extract_proto(self,
                      proto_masks,
                      pos_rois,
                      pos_embeds,
                      stride=4):
        """
        :param proto_masks: proto_masks of 1/4 scales
        :param pos_rois:    the gt_rois sampled by sampler.
        :param pos_embeds:  the pos_embeds produced by bbox_head.
        :param cfg: configs, that tells the strides of rois
        :return: a list of proto_inst_rois
        """
        assert pos_rois.size(0) == pos_embeds.size(0)
        rois_img_inds = pos_rois[:, 0].detach().long()
        # [n, in_embeds, H, W]
        rois_proto_masks = proto_masks.index_select(dim=0, index=rois_img_inds)
        in_embeds, map_h, map_w = rois_proto_masks.shape[1:]
        rois_proto_masks = rois_proto_masks.permute(0, 2, 3, 1).contiguous().view(
            rois_proto_masks.shape[0], map_h * map_w, in_embeds)
        # [n, in_embeds, 1]
        pos_embeds = pos_embeds[..., None]
        # rois_binary_masks: [n, H * W, 1] -> [n, H, W]
        rois_binary_masks = torch.matmul(rois_proto_masks, pos_embeds).view(
            rois_proto_masks.shape[0], map_h, map_w)
        binary_rois = []        # each roi mask: [1, roi_h, roi_w]
        pos_proposals = (pos_rois[:, 1:].detach() / stride).long()
        for i, bbox in enumerate(pos_proposals):
            b_mask = rois_binary_masks[i]
            x1, y1, x2, y2 = bbox
            w = max(x2 - x1 + 1, 1)
            h = max(y2 - y1 + 1, 1)
            binary_rois.append(
                b_mask[y1:y1 + h, x1:x1 + w].unsqueeze(0))
        if not binary_rois:
            binary_rois = pos_rois.new_zeros((0, 0, 0))
        return binary_rois

    def get_target(self, sampling_results, gt_masks, stride=4):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = self.mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks,
                                        stride)
        return mask_targets

    def mask_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_masks, stride):
        num_pos = pos_proposals.size(0)
        rois_gt_targets = []
        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            pos_bboxes = (pos_proposals.detach() / stride).long()
            pos_bboxes = pos_bboxes.cpu().numpy()
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / stride, interpolation='nearest')

                bbox = pos_bboxes[i]
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                target = gt_mask[y1:y1 + h, x1:x1 + w][None, ...]
                target = torch.from_numpy(target).float().to(pos_proposals.device)
                rois_gt_targets.append(target)
        else:
            # rois_gt_targets.append(pos_proposals.new_zeros((0, 0, 0)))
            rois_gt_targets = pos_proposals.new_zeros((0, 0, 0))
        return rois_gt_targets

    def mask_target(self, pos_proposal_list, pos_assigned_gt_inds_list, gt_masks_list,
                    stride):
        stride_list = [stride for _ in range(len(pos_proposal_list))]
        mask_target_list = map(self.mask_target_single, pos_proposal_list,
                           pos_assigned_gt_inds_list, gt_masks_list, stride_list)
        mask_targets = []
        for l in mask_target_list:
            mask_targets.extend(l)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def mask_loss(self, mask_pred, mask_targets):
        loss = dict()
        assert len(mask_pred) == len(mask_targets)
        losses_list = []
        for pred_i, target_i in zip(mask_pred, mask_targets):
            loss_i = F.binary_cross_entropy_with_logits(
                pred_i, target_i.float(), reduction='mean')[None]
            losses_list.append(loss_i)
        if losses_list:
            losses_list = torch.cat(losses_list)
        else:
            losses_list = mask_pred.new_zeros((0, ))
        loss_mask = torch.mean(losses_list)
        loss['loss_mask'] = loss_mask
        return loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # change the neck ...
        proto, x = self.extract_feat(img)
        losses = dict()

        cls_scores, bbox_preds, bbox_embeds = self.bbox_head(x)
        outs = (cls_scores, bbox_preds)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas,
                              self.train_cfg.bbox_head)
        loss_bbox = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(loss_bbox)

        # proposal
        proposal_cfg = self.train_cfg.get('proposals')
        proposal_inputs = outs + (bbox_embeds, img_metas, proposal_cfg)
        proposal_list = self.bbox_head.get_proposals(*proposal_inputs)
        proposal_bbox, proposal_labels, proposal_embeds = map(list, zip(*proposal_list))

        # assign gts and sample proposals
        assigner = build_assigner(self.train_cfg.mask_head.assigner)
        sampler  = build_sampler(self.train_cfg.mask_head.sampler,
                                 context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = assigner.assign(
                proposal_bbox[i],
                gt_bboxes[i],
                gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = sampler.sample(
                assign_result,
                proposal_bbox[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # mask_rois
        pos_rois = bbox2roi(
            res.pos_bboxes for res in sampling_results)
        pos_gt_rois = bbox2roi(
            res.pos_gt_bboxes for res in sampling_results)
        pos_embed_list = [
            proposal_embeds[i][res.pos_inds] for i, res in enumerate(sampling_results)]
        pos_embeds = torch.cat(pos_embed_list)
        assert pos_embeds.size(0) == pos_gt_rois.size(0)

        # crop the rois from proto_masks, then calculate the loss
        # using pos_bboxes but not pos_gt_rois
        mask_targets = self.get_target(sampling_results, gt_masks,
                                       stride=self.train_cfg.proto.stride)
        mask_preds   = self.extract_proto(proto, pos_rois, pos_embeds,
                                          stride=self.train_cfg.proto.stride)
        loss_mask = self.mask_loss(mask_preds, mask_targets)
        losses.update(loss_mask)
        return losses

    def simple_test_mask(self,
                         proto_pred,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         det_embeds,
                         cfg,
                         rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            inst_mask_results = [[] for _ in range(self.bbox_head.num_classes)]
        else:
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])

            assert _bboxes.shape[0] == det_embeds.shape[0]
            # a list of inst_masks.
            mask_preds = self.extract_proto(proto_pred, mask_rois, det_embeds,
                                            stride=cfg.stride)
            # get inst masks from
            inst_mask_results = self.get_inst_masks(mask_preds, _bboxes, det_labels,
                                                    cfg, ori_shape, scale_factor,
                                                    rescale)
        return inst_mask_results

    def get_inst_masks(self, inst_preds, det_bboxes, det_labels, proto_test_cfg,
                      ori_shape, scale_factor, rescale):
        """
        :param proto_pred: a list of instance mask preds.
        :param det_bboxes:  bboxes.
        :param det_labels:
        :param proto_test_cfg:
        :param ori_shape:
        :param scale_factor:
        :param rescale:
        :return:
        """
        inst_preds_np = []
        for inst_pred in inst_preds:
            if isinstance(inst_pred, torch.Tensor):
                inst_pred = inst_pred.sigmoid().cpu().numpy()
                inst_pred = inst_pred.astype(np.float32)
            assert isinstance(inst_pred, np.ndarray)
            inst_preds_np.append(inst_pred)
        # collect the segms by class labels.
        cls_segms = [[] for _ in range(self.bbox_head.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0
        
        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            mask_pred_ = inst_preds_np[i][0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            # from 4x upsample to 1x.
            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            # here to use sigmoid...
            bbox_mask = (bbox_mask > proto_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms

    def simple_test(self, img, img_meta, rescale=False):
        # simple_test, proto is of 1/4 scales.
        proto, x = self.extract_feat(img)
        cls_scores, bbox_preds, bbox_embeds = self.bbox_head(x)
        outs = (cls_scores, bbox_preds)
        bbox_inputs = outs + (bbox_embeds, img_meta, self.test_cfg.bbox_head)
        #
        bbox_list = self.bbox_head.get_proposals(*bbox_inputs,
                                                 rescale=rescale)
        det_bboxes, det_labels, det_embeds = map(list, zip(*bbox_list))
        det_bboxes = torch.cat(det_bboxes)
        det_labels = torch.cat(det_labels)
        det_embeds = torch.cat(det_embeds)
        # use det_bboxes to produce instance masks.
        """use the det_bboxes to crop the instance masks, and use the embeds 
        to multiply the masks and then give out the results.
        """
        # bbox_results = [
        #     bbox2result(det_bboxes_i, det_labels_i, self.bbox_head.num_classes)
        #     for det_bboxes_i, det_labels_i in zip(det_bboxes, det_labels)
        # ]
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        segm_results = self.simple_test_mask(proto, img_meta, det_bboxes, det_labels,
                                             det_embeds, self.test_cfg.proto,
                                             rescale=rescale)
        return bbox_results, segm_results

    def aut_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError






