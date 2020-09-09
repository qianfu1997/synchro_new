# python tools/test.py configs/yolact/mask_rcnn_r50_fpn_1x.py ./work_dirs/paoptic_coco_mrcnn_r50_v1/epoch_12.pth --show
# bash ./tools/dist_test.sh configs/yolact/mask_rcnn_r50_fpn_1x.py work_dirs/panoptic_coco_mrcnn_r50_v2/epoch_12.pth 8 --out results.pkl --eval bbox segm
# bash ./tools/dist_test.sh configs/yolact/mask_rcnn_r50_fpn_1x_plus.py work_dirs/panoptic_coco_mrcnn_r50_plus_v1/epoch_12.pth 8 --out results.pkl --eval bbox segm
bash ./tools/dist_test.sh configs/panoptic/mask_rcnn_r50_fpn_1x_plus.py work_dirs/panoptic_coco_mrcnn_r50_v6/epoch_12.pth 8 --out results.json --