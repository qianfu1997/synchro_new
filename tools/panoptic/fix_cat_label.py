#
# @author:charlotte.Song
# @file: fix_cat_label.py
# @Date: 2019/7/25 14:31
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import os
import os.path as osp
import json
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='fix cat label and gt labels')
    parser.add_argument('ori_category_json', type=str)
    parser.add_argument('save_json', type=str)
    args = parser.parse_args()
    return args

def fix(ori_cat, save_json):
    assert osp.isfile(ori_cat)
    with open(ori_cat, 'r', encoding='utf-8') as f:
        cat_anns = json.loads(f.read(), object_pairs_hook=OrderedDict)

    cat_anns.append({
        'id': 183,
        'isthing': 0,
        'name': 'thing_other'})
    cat_infos = {}

    stuff_anns = []
    thing_anns = []
    for ann in cat_anns:
        if ann['isthing'] == 0:
            stuff_anns.append(ann)
        else:
            thing_anns.append(ann)
    cat_ids = [ann['id'] for ann in cat_anns]
    cat_stuff_ids = [ann['id'] for ann in stuff_anns]
    cat_thing_ids = [ann['id'] for ann in thing_anns]
    cat_infos['cat_ids'] = cat_ids
    cat_infos['cat_stuff_ids'] = cat_stuff_ids
    cat_infos['cat_thing_ids'] = cat_thing_ids
    cat2label = {
        cat_id: i + 1
        for i, cat_id in enumerate(cat_ids)
    }
    cat_infos['cat2label'] = cat2label
    label2cat = {
        i + 1: cat_id
        for i, cat_id in enumerate(cat_ids)
    }
    cat_infos['label2cat'] = label2cat
    catid2name = {
        ann['id']: ann['name']
        for ann in cat_anns
    }
    cat_infos['catid2name'] = catid2name
    gtid2name = {
        cat2label[ann['id']]: ann['name']
        for ann in cat_anns
    }
    cat_infos['gtid2name'] = gtid2name
    stuffcat2stufflabel = {
        ann['id']: i + 1
        for i, ann in enumerate(stuff_anns)
    }
    cat_infos['stuffcat2stufflabel'] = stuffcat2stufflabel
    stufflabel2stuffcat = {
        i + 1: ann['id']
        for i, ann in enumerate(stuff_anns)
    }
    cat_infos['stufflabel2stuffcat'] = stufflabel2stuffcat
    thingcat2thinglabel = {
        ann['id']: i + 1
        for i, ann in enumerate(thing_anns)
    }
    cat_infos['thingcat2thinglabel'] = thingcat2thinglabel
    thinglabel2thingcat = {
        i + 1: ann['id']
        for i, ann in enumerate(thing_anns)
    }
    cat_infos['thinglabel2thingcat'] = thinglabel2thingcat

    with open(save_json, 'w+', encoding='utf-8') as f:
        json.dump(OrderedDict(cat_infos), f)


if __name__ == '__main__':
    args = parse_args()
    fix(args.ori_category_json, args.save_json)