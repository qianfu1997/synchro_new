#
# @author:charlotte.Song
# @file: panoptic_test.py
# @Date: 2019/9/30 22:53
# @description:
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
""" a new test file for panoptic """
import argparse
import os
import os.path as osp
import shutil
import tempfile
import cv2

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model, tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import json
import numpy as np
import matplotlib
""" Panoptic test, collect the segm pngs, annotation dict.
During collecting step, save the pngs in corresponding path,
and then save the annotation 
"""


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def save_png(result_path, png, img, img_meta, **kwargs):
    """Get the result png, and then save the png under the panoptic folder"""
    # TODO
    # if not osp.isdir(result_path):
    #     os.makedirs(result_path)
    # for png, img_meta in zip(pngs, img_metas[0]):
    #     filename = img_meta['filename'].replace('.jpg', '.png')
    #     cv2.imwrite(osp.join(result_path, filename), png)
    # return
    img_meta = img_meta[0].data[0][0]
    if not osp.isdir(result_path):
        os.makedirs(result_path)
    filename = img_meta['filename'].replace('.jpg', '.png')
    filename = osp.split(filename)[-1]
    # cv2.imwrite(osp.join(result_path, filename), png)
    matplotlib.image.imsave(osp.join(result_path, filename), png)
    return


def show_result(data, result_png, show_path):
    # TODO
    """
    首先获取pano png上所有独特的id，然后逐个赋予一个随机的颜色，将之贴上原图。
    """
    if not osp.isdir(show_path):
        os.makedirs(show_path)
    img_tensor = data['img'][0]
    img_metas = data['img_meta'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas) == 1
    # result anns are single dict objects.
    img = imgs[0]
    img_meta = img_metas[0]
    filename = img_meta['filename'].replace('.jpg', '.png')
    filename = osp.split(filename)[-1]
    h, w, _ = img_meta['img_shape']
    img_show = img[:h, :w, :]
    img_show = img_show * 0.5 + result_png * 0.5
    cv2.imwrite(osp.join(show_path, filename), img_show)
    return


# add result_path.
def single_gpu_test(model, data_loader, result_path, show=False,
                    show_path=None):
    """
    :param model:
    :param data_loader:
    :param result_path:
    :param show:
    :return:
        results returned by model contain pngs and anns(dict)
    """
    model.eval()
    annotations = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    pano_result_anns = dict()
    #
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result contains: panoptic_pngs, panoptic_dict.
            # result = model(return_loss=False, rescale=not show, **data)
            result_pngs, result_anns = model(return_loss=False, rescale=True, **data)
            # can be more than one pngs.
            save_png(result_path, result_pngs, **data)
            if show:
                show_result(data, result_pngs, show_path)
        annotations.append(result_anns)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    pano_result_anns['annotations'] = annotations
    return pano_result_anns


def multi_gpu_test(model, data_loader, result_path, show_path, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=True, **data)
            # fist pngs then anns.
            result_pngs, result_anns = model(return_loss=False, rescale=True, **data)
            save_png(result_path, result_pngs, **data)
        # results.append(result)
        results.append(result_anns)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    # results become a list.
    results = collect_results(results, len(dataset), tmpdir)
    return results


def list2json(list_result):
    outputs = dict()
    outputs['annotations'] = list_result
    return outputs


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    if rank != 0:
        return None
    else:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    # do not eval.
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     # choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
    #     choices=['panoptic'],
    #     help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # add argument the panoptic png folders.
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--show_path', type=str, default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # if 'CLASSES' in checkpoint['meta']:
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # outputs is the dict of result anns.
        outputs = single_gpu_test(model, data_loader, args.result_path, args.show,
                                  args.show_path)
    else:
        model = MMDistributedDataParallel(model.cuda())
        # for multi-gpu-test, the outputs is the list of anns.
        # change the list into dict, and then call panoptic api to test the results
        outputs = multi_gpu_test(model, data_loader, args.result_path, args.tmpdir)

    rank, _ = get_dist_info()
    if args.json_out and rank == 0:
        json_name = args.json_out + '.json'
        print('\nwriting results to {}'.format(json_name))
        # args.out must be a json file.
        # needs another function to change
        # if outputs is list, then change it to dict
        if isinstance(outputs, list):
            print(len(outputs))
            outputs = list2json(outputs)
        # mmcv.dump(outputs, args.out)
        with open(json_name, 'w+', encoding='utf-8') as f:
            json.dump(outputs, f)

    # save predictions in the COCO json format.
    # in this way it produce the results by dataset order.
    # if args.json_out and rank == 0:
    #     if not isinstance(outputs[0], dict):
    #         results2json(dataset, outputs, args.json_out)
    #     else:
    #         for name in outputs[0]:
    #             outputs_ = [out[name] for out in outputs]
    #             result_file = args.json_out + '.{}'.format(name)
    #             results2json(dataset, outputs_, result_file)

if __name__ == '__main__':
    main()

