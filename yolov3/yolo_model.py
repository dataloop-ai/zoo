import argparse
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from .test import test  # import test.py to get mAP after each epoch
import tqdm
import time
import glob
import numpy as np
from .model import *
from .yolo_utils.datasets import *
from .yolo_utils.utils import *
from .yolo_utils.parse_config import *



class YoloModel:
    def __init__(self, device):
        this_path = os.path.join(os.getcwd(), 'zoo/yolov3')
        cfg_path = 'cfg/yolov3-custom.cfg'
        weights_path = "weights/darknet53.conv.74"
        # weights_path = ""
        # weights_path = "weights/ultralytics49.pt"

        self.device = device
        self.config_path = os.path.join(this_path, cfg_path)
        self.weights_path = os.path.join(this_path, weights_path)
        self.last_checkpoint_path = os.path.join(this_path, 'weights', 'last.pt')
        self.best_checkpoint_path = os.path.join(this_path, 'weights', 'best.pt')
        self.results_path = os.path.join(this_path, 'weights', 'results.txt')

        self.results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'

        self.cutoff = -1  # backbone reaches to cutoff layer
        self.start_epoch = 0
        self.best_fitness = float('inf')
        self.tb_writer = None

    def train(self, data_dict, other_hyp, epochs=273, batch_size=16, accumulate=4,
              multi_scale=True, img_size=416, rect=False, resume=False, transfer=False, save=True,
              img_weights=False, cache_images=False, arc='default',
              pretrain_bias=False, adam=False):  # effective bs = batch_size * accumulate = 16 * 4 = 64

        self.pretrain_bias = pretrain_bias
        self.save = save

        self.weights_path = self.last_checkpoint_path if resume else self.weights_path
        device = torch_utils.select_device(self.device, batch_size=batch_size)
        other_hyp['obj'] *= img_size / 416.  # scale other_hyp['obj'] by img_size (evolved at 416)


        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter()

        if pretrain_bias:
            self.prebias()
        epochs = 1 if self.pretrain_bias else epochs  # 500200 batches at bs 64, 117263 images = 273 epochs

        if 'pw' not in arc:  # remove BCELoss positive weights_path
            other_hyp['cls_pw'] = 1.
            other_hyp['obj_pw'] = 1.

        # Initialize
        init_seeds()
        if multi_scale:
            img_sz_min = round(img_size / 32 / 1.5)
            img_sz_max = round(img_size / 32 * 1.5)
            img_size = img_sz_max * 32  # initiate with maximum multi_scale size
            print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

        # Configure run
        path_to_imgs_paths_train_file = data_dict['train']
        num_classes = int(data_dict['classes'])  # number of classes

        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob(self.results_path):
            os.remove(f)

        # Initialize model

        # Optimizer
        pg0, pg1 = [], []  # optimizer parameter groups

        self.model = Darknet(self.config_path, arc=arc).to(device)
        for k, v in dict(self.model.named_parameters()).items():
            if 'Conv2d.weight' in k:
                pg1 += [v]  # parameter group 1 (apply weight_decay)
            else:
                pg0 += [v]  # parameter group 0
        self.optimizer = []
        if adam:
            self.optimizer = optim.Adam(pg0, lr=other_hyp['lr0'])
            # optimizer = AdaBound(pg0, lr=other_hyp['lr0'], final_lr=0.1)
        else:
            self.optimizer = optim.SGD(pg0, lr=other_hyp['lr0'], momentum=other_hyp['momentum'], nesterov=True)
        self.optimizer.add_param_group(
            {'params': pg1, 'weight_decay': other_hyp['weight_decay']})  # add pg1 with weight_decay
        del pg0, pg1

        # https://github.com/alphadl/lookahead.pytorch
        # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)
        self._load_model(self.weights_path)

        if transfer or self.pretrain_bias:  # transfer learning edge (yolo) layers
            nf = int(self.model.module_defs[self.model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

            if self.pretrain_bias:
                for p in self.optimizer.param_groups:
                    # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                    p['lr'] *= 100  # lr gain
                    if p.get('momentum') is not None:  # for SGD but not Adam
                        p['momentum'] *= 0.9

            for p in self.model.parameters():
                if self.pretrain_bias and p.numel() == nf:  # train (yolo biases)
                    p.requires_grad = True
                elif transfer and p.shape[0] == nf:  # train (yolo biases+weights_path)
                    p.requires_grad = True
                else:  # freeze layer
                    p.requires_grad = False

        scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[round(epochs * x) for x in [0.8, 0.9]], gamma=0.1)
        scheduler.last_epoch = self.start_epoch - 1

        # Dataset
        dataset = LoadImagesAndLabels(path_to_imgs_paths_train_file,
                                      img_size,
                                      batch_size,
                                      augment=True,
                                      hyp=other_hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      image_weights=img_weights,
                                      cache_labels=True if epochs > 10 else False,
                                      cache_images=False if self.pretrain_bias else cache_images)

        # Dataloader
        batch_size = min(batch_size, len(dataset))
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using %g dataloader workers' % nw)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=min(
                                                     [os.cpu_count(), batch_size if batch_size > 1 else 0, 16]),
                                                 shuffle=not rect,  # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

        # Start training
        self.model.nc = num_classes  # attach number of classes to model
        self.model.arc = arc  # attach yolo architecture
        self.model.hyp = other_hyp  # attach hyperparameters to model
        torch_utils.model_info(self.model, report='summary')  # 'full' or 'summary'
        nb = len(dataloader)

        t0 = time.time()
        print('Starting %s for %g epochs...' % ('pretrain_bias' if self.pretrain_bias else 'training', epochs))
        for epoch in range(self.start_epoch,
                           epochs):  # epoch ------------------------------------------------------------------
            self.model.train()
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

            # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
            freeze_backbone = False
            if freeze_backbone and epoch < 2:
                for name, p in self.model.named_parameters():
                    if int(name.split('.')[1]) < self.cutoff:  # if layer < 75
                        p.requires_grad = False if epoch == 0 else True

            mloss = torch.zeros(4).to(device)  # mean losses
            pbar = tqdm(enumerate(dataloader), total=nb)
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device)
                targets = targets.to(device)

                # Multi-Scale training
                if multi_scale:
                    if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                        img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                    sf = img_size / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / 32.) * 32 for x in
                              imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Run model
                pred = self.model(imgs)

                # Compute loss
                loss, loss_items = compute_loss(pred, targets, self.model)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return self.results

                # Scale loss by nominal batch_size of 64
                loss *= batch_size / 64

                loss.backward()

                # Accumulate gradient for x batches before optimizing
                if ni % accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
                pbar.set_description(s)

                # end batch ------------------------------------------------------------------------------------------------

            # Update scheduler
            scheduler.step()

            # Process epoch results
            self.final_epoch = epoch + 1 == epochs
            if self.pretrain_bias:
                print_model_biases(self.model)
            else:
                # Calculate mAP
                if self.save or self.final_epoch:
                    with torch.no_grad():
                        self.results, maps = test(self.device,
                                                  self.config_path,
                                                  data_dict,
                                                  batch_size=batch_size,
                                                  img_size=img_size,
                                                  model=self.model,
                                                  conf_thres=0.001 if self.final_epoch and epoch > 0 else 0.1,
                                                  # 0.1 for speed
                                                  save_json=self.final_epoch and epoch > 0)
            mAP = self.get_metrics()
            self._write_to_tensorboard(self.results[2], mloss[3], epoch)
            if self.save:
                self._save_checkpoint(self.results, epoch, s)
            # end epoch ----------------------------------------------------------------------------------------------------

        # end training
        plot_results()  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - self.start_epoch + 1, (time.time() - t0) / 3600))
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()

    def get_best_checkpoint(self):
        return torch.load(self.best_checkpoint_path)

    def get_metrics(self):
        return self.results[2]

    def prebias(self):
        # trains output bias layers for 1 epoch and creates new backbone
        if self.pretrain_bias:
            a = self.pretrain_bias  # save settings
            img_weights = False  # disable settings

            self.train()  # transfer-learn yolo biases for 1 epoch
            create_backbone(self.last_checkpoint_path)  # saved results as backbone.pt

            weights = self.weights_dir + 'backbone.pt'  # assign backbone
            prebias = False  # disable pretrain_bias
            img_weights = a  # reset settings

    def _write_to_tensorboard(self, results, mloss, epoch):
        # Write Tensorboard results
        if self.tb_writer:
            x = [mloss.item()] + [results.item()]
            titles = ['Train Loss', '0.5AP']
            for xi, title in zip(x, titles):
                self.tb_writer.add_scalar(title, xi, epoch)

        # if self.tb_writer:
        #     x = list(mloss) + list(results)
        #     titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
        #               'Precision', 'Recall', '0.5AP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
        #     for xi, title in zip(x, titles):
        #         self.tb_writer.add_scalar(title, xi, epoch)

    def _save_checkpoint(self, results, epoch, s):

        with open(self.results_path, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Update best mAP or lowest val GIOU, Objectness and Classification
        fitness = self.results[2]  # get mAP
        if fitness > self.best_fitness:
            self.best_fitness = fitness

        # Save training results

        with open(self.results_path, 'r') as f:
            # Create checkpoint
            checkpoint = {'epoch': epoch,
                          'best_fitness': self.best_fitness,
                          'training_results': f.read(),
                          'model': self.model.state_dict(),
                          'optimizer': None if self.final_epoch else self.optimizer.state_dict()}

            # Save last checkpoint
            torch.save(checkpoint, self.last_checkpoint_path)

            # Save best checkpoint
            if self.best_fitness == fitness:
                torch.save(checkpoint, self.best_checkpoint_path)

            # Delete checkpoint
            del checkpoint

    def _load_model(self, weights_path):
        attempt_download(weights_path)
        if weights_path.endswith('.pt'):  # pytorch format
            # possible weights_path are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            checkpoint = torch.load(weights_path, map_location=self.device)

            # load model
            checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                                   self.model.state_dict()[k].numel() == v.numel()}
            self.model.load_state_dict(checkpoint['model'], strict=False)

            # load optimizer
            if checkpoint['optimizer'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_fitness = checkpoint['best_fitness']

            # load results
            if checkpoint.get('training_results') is not None:
                with open(self.results_path, 'w') as file:
                    file.write(checkpoint['training_results'])  # write results.txt

            self.start_epoch = checkpoint['epoch'] + 1

        elif len(weights_path.split('/')[-1]) > 0:  # darknet format
            # possible weights_path are '*.weights_path', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            self.cutoff = load_darknet_weights(self.model, weights_path)