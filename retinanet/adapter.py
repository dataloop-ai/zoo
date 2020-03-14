import os
from dl_to_csv import create_annotations_txt
from .retinanet_model import RetinaModel
from .visualize import detect

def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

class AdapterModel:

    def __init__(self, devices, model_specs, hp_values):
        self.model_specs = model_specs
        self.annotation_type = model_specs['data']['annotation_type']
        self.hp_values = hp_values
        self.path = os.getcwd()
        self.output_path = os.path.join(self.path, 'output')
        self.training_configs = self.model_specs['training_configs']
        self.classes_filepath = None
        self.annotations_train_filepath = None
        self.annotations_val_filepath = None
        self.home_path = None
        try:
            past_trial_id = self.hp_values['tuner/past_trial_id']
        except:
            past_trial_id = None
        try:
            new_trial_id = self.hp_values['tuner/new_trial_id']
        except:
            new_trial_id = generate_trial_id()
        try:
            resume = self.hp_values['tuner/initial_epoch'] > 0
        except:
            resume = False
        if self.annotation_type == 'coco':
            self.home_path = self.model_specs['data']['home_path']
            self.dataset_name = self.model_specs['data']['dataset_name']
        elif self.annotation_type == 'csv' or self.annotation_type == 'dataloop':
            self.classes_filepath = os.path.join(self.output_path, 'classes.txt')
            self.annotations_train_filepath = os.path.join(self.output_path, 'annotations_train.txt')
            self.annotations_val_filepath = os.path.join(self.output_path, 'annotations_val.txt')
        self.retinanet_model = RetinaModel(devices['gpu_index'], resume, new_trial_id, past_trial_id, self.home_path)

    def reformat(self):
        if self.annotation_type == 'coco':
            pass
        elif self.annotation_type == 'csv':
            pass
        elif self.annotation_type == 'dataloop':
            # convert dataloop annotations to csv styled annotations
            labels_list = self.model_specs['data']['labels_list']
            local_labels_path = os.path.join(self.path, self.model_specs['data']['labels_relative_path'])
            local_items_path = os.path.join(self.path, self.model_specs['data']['items_relative_path'])

            create_annotations_txt(annotations_path=local_labels_path,
                                   images_path=local_items_path,
                                   train_split=0.9,
                                   train_filepath=self.annotations_train_filepath,
                                   val_filepath=self.annotations_val_filepath,
                                   classes_filepath=self.classes_filepath,
                                   labels_list=labels_list)
            self.annotation_type == 'csv'

    def preprocess(self):
        self.retinanet_model.preprocess(dataset=self.annotation_type,
                                        csv_train=self.annotations_train_filepath,
                                        csv_val=self.annotations_val_filepath,
                                        csv_classes=self.classes_filepath,
                                        coco_path=self.home_path,
                                        train_set_name='train' + self.dataset_name,
                                        val_set_name='val' + self.dataset_name,
                                        resize=self.training_configs['input_size'])

    def build(self):
        self.retinanet_model.build(depth=self.training_configs['depth'],
                                   learning_rate=self.hp_values['learning_rate'],
                                   ratios=self.hp_values['anchor_ratios'],
                                   scales=self.hp_values['anchor_scales'])

    def train(self):
        self.retinanet_model.train(epochs=self.hp_values['tuner/epochs'],
                                   init_epoch=self.hp_values['tuner/initial_epoch'])

    def get_checkpoint(self):
        return self.retinanet_model.get_best_checkpoint()

    def get_metrics(self):
        return {'val_accuracy': self.retinanet_model.get_metrics().item()}


def predict(home_path, checkpoint_path):
    class_names_path = os.path.join(home_path, "d.names")
    num_classes = sum(1 for line in open(class_names_path))
    predict_on_path = os.path.join(home_path, 'predict_on')
    detect(csv_classes=class_names_path, pred_on_path=predict_on_path, checkpoint_path=checkpoint_path,
           num_classes=num_classes, ouptut_path=os.path.join(home_path, 'predicted_on'))
