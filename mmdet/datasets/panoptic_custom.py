#
# @author:charlotte.Song
# @file: panoptic_custom.py
# @Date: 2019/9/18 20:53
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS
"""Needs to rewrite the pipelines for panoptic segmentation."""


@DATASETS.register_module
class PanopticCustomDataset(Dataset):
    """Custon dataset for panoptic
    Instance Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height: 720,
            'ann': {
                'bboxes': <np.ndrray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, )}
        },
        ...
    ]
    Except for instance, also parse stuff semantic segmentation annotations.
    PNGs Annotations that label pixels with their cat id, need to be transfered
    to stuff_label for training.

    For training pipelines, add cat_infos into results.
    """
    CLASSES = None
    STUFF_CLASSES = None
    THING_CLASSES = None

    def __init__(self,
                 ann_file,   # for instance
                 category_file,  # for panoptic
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        self.ann_file = ann_file
        self.categroy_file = category_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations
        # just load the instance information
        self.img_infos = self.load_annotations(self.ann_file)
        self.cat_infos = self.load_category_anns(self.categroy_file)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images with no annotation during training
        # for training then use this filtering methods.
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        if not self.test_mode:
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_category_anns(self, category_file):
        return mmcv.load(category_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        # add cat_infos for sequential training pipeline.
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small. """
        valid_inds = []
        for i, img_info in enumerate(self.img_info):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            # data ==
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        # here to add the cat_infos
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       cat_info=self.cat_infos)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info,
                       cat_info=self.cat_infos,
                       # image id.
                       id=img_info['id'])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)