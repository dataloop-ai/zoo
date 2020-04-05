import numpy as np
import time
import os
import skimage
import cv2
from . import model
import torch
from .utils import combine_values
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataloaders import PredDataset, collater, Resizer, Normalizer, UnNormalizer
try:
    from logging_utils import logginger
    logger = logginger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


def detect(checkpoint, output_dir, visualize=False):

    home_path = checkpoint['model_specs']['data']['home_path']
    if os.getcwd().split('/')[-1] == 'zoo':
        home_path = os.path.join('..', home_path)
    # must have a file to predict on called "predict_on"
    pred_on_path = os.path.join(home_path, 'predict_on')

    #create output path
    output_path = os.path.join(home_path, 'predictions', output_dir)
    if not os.path.exists(os.path.join(home_path, 'predictions')):
        os.mkdir(os.path.join(home_path, 'predictions'))
    if os.path.exists(output_path):
        raise Exception('there are already predictions for model: ' + output_dir)
    os.mkdir(output_path)
    try:
        logger.info('inside ' + str(pred_on_path) + ': ' + str(os.listdir(pred_on_path)))
        dataset_val = PredDataset(pred_on_path=pred_on_path,
                                  transform=transforms.Compose([Normalizer(), Resizer(min_side=608)])) #TODO make resize an input param

        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=None)

        labels = checkpoint['labels']
        num_classes = len(labels)

        configs = combine_values(checkpoint['model_specs']['training_configs'], checkpoint['hp_values'])

        retinanet = model.resnet152(num_classes=num_classes, scales=configs['anchor_scales'], ratios=configs['anchor_ratios']) #TODO: make depth an input parameter
        retinanet.load_state_dict(checkpoint['model'])
        retinanet = retinanet.cuda()
        retinanet.eval()
        unnormalize = UnNormalizer()

        def draw_caption(image, box, caption):
            b = np.array(box).astype(int)
            cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        for idx, data in enumerate(dataloader_val):
            scale = data['scale'][0]
            with torch.no_grad():
                st = time.time()
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
                print('Elapsed time: {}'.format(time.time() - st))
                idxs = np.where(scores.cpu() > 0.5)[0]
                if visualize:
                    img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                    img[img < 0] = 0
                    img[img > 255] = 255

                    img = np.transpose(img, (1, 2, 0))
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                detections_list = []
                for j in range(idxs.shape[0]):
                    bbox = transformed_anchors[idxs[j], :]
                    if visualize:
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])

                    label_idx = int(classification[idxs[j]])
                    label_name = labels[label_idx]
                    score = scores[idxs[j]].item()
                    if visualize:
                        draw_caption(img, (x1, y1, x2, y2), label_name)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                        print(label_name)

                    # un resize for eval against gt
                    bbox /= scale
                    bbox.round()
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    detections_list.append([label_name, str(score), str(x1), str(y1), str(x2), str(y2)])
                img_name = dataset_val.image_names[idx].split('/')[-1]
                i_name = img_name.split('.')[0]
                filename = i_name + '.txt'
                filepathname = os.path.join(output_path, filename)
                with open(filepathname, 'w', encoding='utf8') as f:
                    for single_det_list in detections_list:
                        for i, x in enumerate(single_det_list):
                            f.write(str(x))
                            f.write(' ')
                        f.write('\n')
                if visualize:
                    save_to_path = os.path.join(output_path, img_name)
                    cv2.imwrite(save_to_path, img)
                    cv2.waitKey(0)
    except:
        os.remove(output_path)

    return output_path

def detect_single_image(checkpoint, image_path, visualize=False):

    configs = combine_values(checkpoint['model_specs']['training_configs'], checkpoint['hp_values'])
    labels = checkpoint['labels']
    num_classes = len(labels)
    retinanet = model.resnet152(num_classes=num_classes, scales=configs['anchor_scales'], ratios=configs['anchor_ratios']) #TODO: make depth an input parameter
    retinanet.load_state_dict(checkpoint['model'])
    retinanet = retinanet.cuda()
    retinanet.eval()

    img = skimage.io.imread(image_path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([Normalizer(), Resizer(min_side=608)]) #TODO: make this dynamic
    data = transform({'img': img, 'annot': np.zeros((0, 5))})
    img = data['img']
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    with torch.no_grad():
        scores, classification, transformed_anchors = retinanet(img.cuda().float())


        idxs = np.where(scores.cpu() > 0.5)[0]
        scale = data['scale']
        detections_list = []
        for j in range(idxs.shape[0]):
            bbox = transformed_anchors[idxs[j], :]
            label_idx = int(classification[idxs[j]])
            label_name = labels[label_idx]
            score = scores[idxs[j]].item()

            # un resize for eval against gt
            bbox /= scale
            bbox.round()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            detections_list.append([label_name, str(score), str(x1), str(y1), str(x2), str(y2)])
        img_name = image_path.split('/')[-1].split('.')[0]
        filename = img_name + '.txt'
        path = os.path.dirname(image_path)
        filepathname = os.path.join(path, filename)
        with open(filepathname, 'w', encoding='utf8') as f:
            for single_det_list in detections_list:
                for i, x in enumerate(single_det_list):
                    f.write(str(x))
                    f.write(' ')
                f.write('\n')

        if visualize:
            unnormalize = UnNormalizer()


    return filepathname

