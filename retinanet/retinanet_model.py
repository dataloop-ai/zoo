import collections
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from . import model
from zoo.dataloaders.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, Normalizer
from torch.utils.data import DataLoader

from . import csv_eval
from . import coco_eval
import logging

logger = logging.getLogger(__name__)

print('CUDA available: {}'.format(torch.cuda.is_available()))


class RetinaModel:
    def __init__(self, device, home_path):
        self.home_path = home_path
        self.device = device
        this_path = os.path.join(os.getcwd(), 'zoo/retinanet')
        self.weights_dir_path = self.last_checkpoint_path = os.path.join(this_path, 'weights')
        self.last_checkpoint_path = os.path.join(this_path, 'weights', 'last.pt')
        self.best_checkpoint_path = os.path.join(this_path, 'weights', 'best.pt')
        self.results_path = os.path.join(this_path, 'weights', 'results.txt')

        self.best_fitness = - float('inf')
        self.tb_writer = None

    def preprocess(self, dataset='csv', csv_train=None, csv_val=None, csv_classes=None, coco_path=None,
                   train_set_name='train2017', val_set_name='val2017', resize=608):
        self.dataset = dataset
        if self.dataset == 'coco':
            if coco_path is None:
                raise ValueError('Must provide --home_path when training on COCO,')
            self.dataset_train = CocoDataset(coco_path, set_name=train_set_name,
                                             transform=transforms.Compose(
                                                 [Normalizer(), Augmenter(), Resizer(min_side=resize)]))
            self.dataset_val = CocoDataset(coco_path, set_name=val_set_name,
                                           transform=transforms.Compose([Normalizer(), Resizer(min_side=resize)]))

        elif self.dataset == 'csv':
            if csv_train is None:
                raise ValueError('Must provide --csv_train when training on COCO,')
            if csv_classes is None:
                raise ValueError('Must provide --csv_classes when training on COCO,')
            self.dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                                            transform=transforms.Compose(
                                                [Normalizer(), Augmenter(), Resizer(min_side=resize)])
                                            )

            if csv_val is None:
                self.dataset_val = None
                print('No validation annotations provided.')
            else:
                self.dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes,
                                              transform=transforms.Compose([Normalizer(), Resizer(min_side=resize)]))
        else:
            raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

        sampler = AspectRatioBasedSampler(self.dataset_train, batch_size=2, drop_last=False)
        self.dataloader_train = DataLoader(self.dataset_train, num_workers=0, collate_fn=collater,
                                           batch_sampler=sampler)
        if self.dataset_val is not None:
            sampler_val = AspectRatioBasedSampler(self.dataset_val, batch_size=1, drop_last=False)
            self.dataloader_val = DataLoader(self.dataset_val, num_workers=3, collate_fn=collater,
                                             batch_sampler=sampler_val)

        print('Num training images: {}'.format(len(self.dataset_train)))

    def build(self, depth=50, learning_rate=1e-5, scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
        # Create the model

        if depth == 18:
            retinanet = model.resnet18(num_classes=self.dataset_train.num_classes(), scales=scales,
                                       weights_dir=self.weights_dir_path,
                                       pretrained=True)
        elif depth == 34:
            retinanet = model.resnet34(num_classes=self.dataset_train.num_classes(), scales=scales,
                                       weights_dir=self.weights_dir_path,
                                       pretrained=True)
        elif depth == 50:
            retinanet = model.resnet50(num_classes=self.dataset_train.num_classes(), scales=scales,
                                       weights_dir=self.weights_dir_path,
                                       pretrained=True)
        elif depth == 101:
            retinanet = model.resnet101(num_classes=self.dataset_train.num_classes(), scales=scales,
                                        weights_dir=self.weights_dir_path,
                                        pretrained=True)
        elif depth == 152:
            retinanet = model.resnet152(num_classes=self.dataset_train.num_classes(), scales=scales,
                                        weights_dir=self.weights_dir_path,
                                        pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
        self.scales = scales
        self.retinanet = retinanet.cuda(device=self.device)
        self.retinanet.training = True
        self.optimizer = optim.Adam(self.retinanet.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def train(self, epochs=100, save=True):

        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter()
        for epoch_num in range(epochs):

            print('total epochs: ', epochs)
            self.retinanet.train()
            self.retinanet.freeze_bn()

            epoch_loss = []
            loss_hist = collections.deque(maxlen=500)
            total_num_iterations = len(self.dataloader_train)
            dataloader_iterator = iter(self.dataloader_train)
            pbar = tqdm(total=total_num_iterations)

            for iter_num in range(1, total_num_iterations + 1):
                try:
                    data = next(dataloader_iterator)
                    self.optimizer.zero_grad()
                    classification_loss, regression_loss = self.retinanet(
                        [data['img'].cuda(device=self.device).float(), data['annot'].cuda(device=self.device)])
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss
                    if bool(loss == 0):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.retinanet.parameters(), 0.1)
                    self.optimizer.step()
                    loss_hist.append(float(loss))
                    epoch_loss.append(float(loss))
                    s = 'Epoch: {}/{} | Iteration: {}/{}  | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, epochs, iter_num, total_num_iterations, float(classification_loss), float(regression_loss), np.mean(loss_hist))
                    pbar.set_description(s)
                    pbar.update()
                    del classification_loss
                    del regression_loss
                except Exception as e:
                    logger.info(e)
                    pbar.update()
                    continue
            pbar.close()
            self.scheduler.step(np.mean(epoch_loss))
            self.final_epoch = epoch_num + 1 == epochs

            mAP = self.get_metrics()
            self._write_to_tensorboard(mAP, np.mean(loss_hist), epoch_num)

            if save:
                self._save_checkpoint(mAP, epoch_num)
            if self.final_epoch:
                self._save_classes_for_inference()

    def get_best_checkpoint(self):
        return torch.load(self.best_checkpoint_path)

    def get_metrics(self):

        mAP = csv_eval.evaluate(self.dataset_val, self.retinanet)
        return mAP

    def save(self):
        torch.save(self.retinanet, 'model_final.pt')

    def _save_classes_for_inference(self):
        classes_path = os.path.join(self.home_path, "d.names")
        if os.path.exists(classes_path):
            os.remove(classes_path)
        print("saving classes to be used later for inference at ", classes_path)
        with open(classes_path, "w") as f:
            for key in self.dataset_train.classes.keys():
                f.write(key)
                f.write("\n")

    def _write_to_tensorboard(self, results, mloss, epoch):

        # Write Tensorboard results
        if self.tb_writer:
            x = [mloss.item()] + [results.item()]
            titles = ['Train_Loss', '0.5AP']
            for xi, title in zip(x, titles):
                self.tb_writer.add_scalar(title, xi, epoch)

    def _save_checkpoint(self, results, epoch):

        # Update best mAP
        fitness = results  # total loss
        if fitness > self.best_fitness:
            self.best_fitness = fitness

        # Create checkpoint
        checkpoint = {'epoch': epoch,
                      'best_fitness': self.best_fitness,
                      'training_results': results,
                      'model': self.retinanet.state_dict(),
                      'optimizer': None if self.final_epoch else self.optimizer.state_dict(),
                      'scales': self.scales}

        # Save last checkpoint
        torch.save(checkpoint, self.last_checkpoint_path)

        # Save best checkpoint
        if self.best_fitness == fitness:
            torch.save(checkpoint, self.best_checkpoint_path)

        # Delete checkpoint
        del checkpoint


if __name__ == '__main__':
    pass
