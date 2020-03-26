import numpy as np
import time
import argparse
import glob
import os
import cv2
from . import model
import torch
from shutil import copyfile
from torch.utils.data import DataLoader
from torchvision import transforms

from zoo.dataloaders.dataloader import CocoDataset, PredDataset , PDataset, collater, Resizer, AspectRatioBasedSampler, \
    UnNormalizer, Normalizer


def detect(home_path, checkpoint_path):

    # must have a file to predict on called "predict_on"
    pred_on_path = os.path.join(home_path, 'predict_on')

    checkpoint_name = checkpoint_path.split('.')[0]
    #create output path
    output_path = os.path.join(home_path, 'predictions', checkpoint_name)
    if not os.path.exists(os.path.join(home_path, 'predictions')):
        os.mkdir(os.path.join(home_path, 'predictions'))
    if os.path.exists(output_path):
        raise Exception('there are already predictions for model: ' + checkpoint_name)
    os.mkdir(output_path)

    class_names_path = os.path.join(home_path, "d.names")
    dataset_val = PredDataset(pred_on_path=pred_on_path, class_list_path=class_names_path,
                              transform=transforms.Compose([Normalizer(), Resizer(min_side=608)])) #TODO make resize an input param

    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=None)

    checkpoint = torch.load(checkpoint_path)
    scales = checkpoint['scales']
    ratios = checkpoint['ratios']

    num_classes = sum(1 for line in open(class_names_path))
    retinanet = model.resnet152(num_classes=num_classes, scales=scales, ratios=ratios) #TODO: make depth an input parameter
    retinanet.load_state_dict(checkpoint['model'])
    retinanet = retinanet.cuda()
    retinanet.eval()

    for idx, data in enumerate(dataloader_val):
        scale = data['scale'][0]
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)[0]

            detections_list = []
            for j in range(idxs.shape[0]):
                bbox = transformed_anchors[idxs[j], :]
                label_idx = int(classification[idxs[j]])
                label_name = dataset_val.labels[label_idx]
                score = scores[idxs[j]].item()

                # un resize for eval against gt
                bbox /= scale
                bbox.round()
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                detections_list.append([label_name, str(score), str(x1), str(y1), str(x2), str(y2)])
            img_name = dataset_val.image_names[idx].split('/')[-1]
            filename = img_name + '.txt'
            filepathname = os.path.join(output_path, filename)
            with open(filepathname, 'w', encoding='utf8') as f:
                for single_det_list in detections_list:
                    for i, x in enumerate(single_det_list):
                        f.write(str(x))
                        f.write(' ')
                    f.write('\n')

    return output_path
