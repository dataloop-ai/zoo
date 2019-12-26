import numpy as np
import time
import argparse
import os
import cv2
from . import model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from zazoo.dataloaders.dataloader import CocoDataset, PredDataset, collater, Resizer, AspectRatioBasedSampler, \
    UnNormalizer, Normalizer


def detect(checkpoint_path, num_classes, pred_on_path, dataset='csv', csv_classes=None, ouptut_path=None, home_path=None):
    dataset_val = PredDataset(pred_on_path=pred_on_path, class_list=csv_classes,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=None)
    checkpoint = torch.load(checkpoint_path)
    try:
        scales = checkpoint['scales']
    except:
        scales = [1, 1.2599210498948732, 1.5874010519681994]

    retinanet = model.resnet50(num_classes=num_classes, scales=scales)
    retinanet.load_state_dict(checkpoint['model'])
    retinanet = retinanet.cuda()
    retinanet.eval()
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_idx = int(classification[idxs[0][j]])
                label_name = dataset_val.labels[label_idx]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)
            img_name = dataset_val.image_names[idx].split('/')[-1]
            save_to_path = os.path.join(ouptut_path, img_name)
            cv2.imwrite(save_to_path, img)
            cv2.waitKey(0)
