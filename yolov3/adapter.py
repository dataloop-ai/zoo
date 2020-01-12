from .yolo_model import YoloModel
from .detect import detect
import os
import subprocess


class AdapterModel:

    def __init__(self, devices, model_specs, hp_values, final):
        self.final = final
        self.path = os.getcwd()
        self.home_path = model_specs['data']['home_path']
        self.training_configs = model_specs['training_configs']
        self.hp_values = hp_values
        self.yolo_model = YoloModel(str(devices['gpu_index']))

    def reformat(self):
        pass

    def data_loader(self):
        pass

    def preprocess(self):
        pass

    def build(self):
        pass

    def train(self):
        # Hyperparameters (results68: 59.2 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310
        hyp = {'giou': 3.31,  # giou loss gain
               'cls': 42.4,  # cls loss gain
               'cls_pw': 1.0,  # cls BCELoss positive_weight
               'obj': 52.0,  # obj loss gain (*=img_size/416 if img_size != 416)
               'obj_pw': 1.0,  # obj BCELoss positive_weight
               'iou_t': 0.213,  # iou training threshold
               'lr0': self.hp_values['learning_rate'],  # initial learning rate (SGD=1E-3, Adam=9E-5)
               'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
               'momentum': self.hp_values['momentum'],  # SGD momentum
               'weight_decay': 0.000489,  # optimizer weight decay
               'fl_gamma': 0.5,  # focal loss gamma
               'hsv_h': 0.0103,  # image HSV-Hue augmentation (fraction)
               'hsv_s': 0.691,  # image HSV-Saturation augmentation (fraction)
               'hsv_v': 0.433,  # image HSV-Value augmentation (fraction)
               'degrees': 1.43,  # image rotation (+/- deg)
               'translate': 0.0663,  # image translation (+/- fraction)
               'scale': 0.11,  # image scale (+/- gain)
               'shear': 0.384}  # image shear (+/- deg)

        class_names_path = os.path.join(self.home_path, "d.names")
        num_classes = sum(1 for line in open(class_names_path))
        if os.path.exists('zoo/yolov3/cfg/yolov3-custom.cfg'):
            os.remove('zoo/yolov3/cfg/yolov3-custom.cfg')

        cmd_line = 'bash ' + 'zoo/yolov3/cfg/create_custom_model.sh ' + str(num_classes)
        os.system(cmd_line)
        data = {
            "train": os.path.join(self.home_path, "train_paths.txt"),
            "valid": os.path.join(self.home_path, "val_paths.txt"),
            "classes": num_classes,
            "names": class_names_path
        }
        self.yolo_model.train(data, other_hyp=hyp,
                              epochs=self.training_configs['epochs'],
                              batch_size=self.training_configs['batch_size'],
                              img_size=self.training_configs['input_size'],
                              save=self.final)

    def get_checkpoint(self):
        return self.yolo_model.get_best_checkpoint()

    def get_metrics(self):
        return {'val_accuracy': self.yolo_model.get_metrics().item()}


def predict(home_path, checkpoint_path):
    class_names_path = os.path.join(home_path, "d.names")
    num_classes = sum(1 for line in open(class_names_path))
    if not os.path.exists('zoo/yolov3/cfg/yolov3-custom.cfg'):
        cmd_line = 'bash ' + 'zoo/yolov3/cfg/create_custom_model.sh ' + str(num_classes)
        os.system(cmd_line)
    data = {
        "predict_on": os.path.join(home_path, "predict_on"),
        "names": class_names_path
    }
    output_path = os.path.join(home_path, "predicted_on")
    detect(data, device='0', checkpoint_path=checkpoint_path, out=output_path)
