# CUDA_VISIBLE_DEVICES=2,3,6,7 bash ./tools/dist_train.sh ./configs/yolact/mask_rcnn_r50_fpn_1x.py 4 --work_dir ./work_dirs/panoptic_coco_mrcnn_r50_v1/
# yolact_plus_v2: FPN_plus
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh ./configs/yolact/mask_rcnn_r50_fpn_1x_plus.py 8 --work_dir ./work_dirs/panoptic_coco_mrcnn_r50_plus_v4/
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh ./configs/panoptic/mask_rcnn_r50_fpn_1x_plus.py 8 --work_dir ./work_dirs/panoptic_coco_mrcnn_r50_v3/
# CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 bash ./tools/dist_train.sh ./configs/panoptic/mask_rcnn_r50_fpn_1x_plus.py 8 --work_dir ./work_dirs/panoptic_coco_mrcnn_r50_v4/
CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 bash ./tools/dist_train.sh ./configs/panoptic/mask_rcnn_r50_fpn_1x_plus.py 8 --work_dir ./work_dirs/panoptic_coco_mrcnn_r50_1x/