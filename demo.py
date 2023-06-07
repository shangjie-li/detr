import argparse
import random
import time
from PIL import Image
from PIL import ImageDraw

import numpy as np
import torch

import util.misc as utils
from datasets.coco import make_coco_transforms
from models import build_model
from main import get_args_parser


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    model.eval()

    transforms = make_coco_transforms('val')
    if args.image is not None:
        img = Image.open(args.image)  # <class 'PIL.Image.Image'>
        w, h = img.size

        start = time.time()
        img_t, _ = transforms(img, None)  # [3, H, W], the image is rescaled to have min size 800 and max size 1333

        # the function of collate_fn:
        # 1. find the max_h and max_w of the images in the batch
        # 2. align all the images to the top left corner, and fill in 0 to resize all images to (max_h, max_w)
        # 3. create mask to indicate which positions are filled with 0
        samples, _ = utils.collate_fn([(img_t, None)])
        samples = samples.to(device)
        # print('type(samples):', type(samples))  # <class 'util.misc.NestedTensor'>
        # print('samples.tensors.shape:', samples.tensors.shape)  # [B, 3, H, W], min(H, W) is 800 or max(H, W) is 1333
        # print('samples.mask.shape:', samples.mask.shape)  # [B, H, W]
        # i_array = samples.tensors[0].permute(1, 2, 0).cpu().numpy()
        # i_array = i_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        # i_pil = Image.fromarray((i_array * 255.0).astype(np.uint8), 'RGB')
        # i_pil.show()

        outputs = model(samples)
        # print(outputs['pred_logits'].shape)  # [B, num_queries, 92]
        # print(outputs['pred_boxes'].shape)  # [B, num_queries, 4], cxcywh format

        orig_target_sizes = torch.from_numpy(np.array([[h, w]])).to(device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        scores = results[0]['scores']  # [num_queries]
        labels = results[0]['labels']  # [num_queries]
        boxes = results[0]['boxes']  # [num_queries, 4], xyxy format
        end = time.time()
        print('Inference time: %.3f s' % (end - start))

        x = ImageDraw.ImageDraw(img)
        for i in range(args.num_queries):
            score = scores[i].item()
            if score < args.score_thresh:
                continue

            label, b = labels[i].item(), boxes[i].tolist()
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            x.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=2)

            label_str = COCO_CLASSES[COCO_LABEL_MAP[int(label)] - 1] + ': %.3f' % score
            x.text((x1 + 4, y1 + 2), label_str, fill=(255, 0, 0))

        img.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR demo script', parents=[get_args_parser()])
    parser.add_argument('--image', default=None, type=str,
                        help='path to the input image')
    parser.add_argument('--score_thresh', default=0.7, type=float,
                        help='score threshold for filtering results')
    args = parser.parse_args()

    if args.resume == '':
        args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'

    if args.image is None:
        raise ValueError('the input image must be set')

    main(args)
