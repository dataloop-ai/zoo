## Getting started with the ***ObjectDetNet***
ObjectDetNet is an easy, flexible, open-source object detection framework which allows you to easily train & resume 
training sessions, run inference and flexibly work with checkpoints for a production grade environment.

At the core of the *ObjectDetNet* framework is the ***checkpoint object***. The ***checkpoint object*** Which is a json or json styled file to be 
loaded into python as a dictionary. Bellow is an example of what a checkpoint object might look like.
```
├── {} devices
│   ├── {} gpu_index
│       ├── 0
├── {} model_specs
│   ├── {} name
│       ├── retinanet
│   ├── {} training_configs
│       ├── {} depth
│           ├── 152
│       ├── {} input_size
│       ├── {} learning_rate
│   ├── {} data
│       ├── {} home_path
│       ├── {} annotation_type
│           ├── coco
│       ├── {} dataset_name
├── {} hp_values
│       ├── {} learning_rate
│       ├── {} tuner/epochs
│       ├── {} tuner/initial_epoch
├── {} labels
│       ├── {} 0
│           ├── Rodent
│       ├── {} 1
│       ├── {} 2
├── {} metrics
│       ├── {} val_accuracy
│           ├── 0.834
├── {} model
├── {} optimizer
├── {} scheduler
├── {} epoch
│       ├── 18
```
The model, optimizer and scheduler keys are necessary for resuming a training session.


## Adding your own model to the ***ObjectDetNet***
We encourage you to add your own model to the *ZazuML model zoo* and become a contributor to the project. 

### Example of the directory structure of your model
```
├── retinanet
│   ├── __init__.py
│   ├── adapter.py
│   ├── anchors.py
│   ├── dataloaders
│   ├── losses.py
│   ├── model.py
│   ├── oid_dataset.py
│   ├── train_model.py
│   ├── utils.py
```
<br/><br/>    

Every model must have a mandatory ***adapter.py*** file which contains an **AdapterModel** 
class and a ***predict*** function, which serves as an adapter between ***ZazuML*** and our ***ZaZoo*** 

### Template for your AdapterModel class
```
class AdapterModel:

    def load(self, checkpoint_path='checkpoint.pt'):
        raise NotImplementedError

    def reformat(self):
        pass

    def preprocess(self, batch):
        return batch

    def build(self):
        pass

    def train(self):
        pass

    def get_checkpoint(self):
        pass

    @property
    def checkpoint_path(self):
        raise NotImplementedError

    def save(self, save_path):
        raise NotImplementedError

    def predict(self, checkpoint_path='checkpoint.pt', output_dir='checkpoint0'):
        raise NotImplementedError

    def predict_single_image(self, image_path, checkpoint_path='checkpoint.pt'):
        raise NotImplementedError
```
The "init", "train", "get_checkpoint" and "get_metrics" methods are mandatory methods for running your model. 
The methods are run in the order of the example above, i.e. first the "init" then "reformat" and so on . . 

**load** method is where you pass all the important information to your model 

- device - gpu index to be specified to all parameters and operations requiring gpu in this specific trial
- model_specs - contains model configurations and information relevant to the location of your data and annotation type
- hp_values - are the final hyper parameter values passed to this specific trial

**reformat** method is where you'd be expected to reformat the input image annotations into a format your
model can handle. Your model is required to handle CSV and Coco styled annotations at the very least.

**get_checkpoint** method is expected to return a pytorch styled .pt file

**get_metrics** method is expected to return a dictionary object in the form of `{'val_accuracy': 0.928}` 
where `0.928` in this example is a python float

**predict** function receives the path to your data storage directory as well as to a checkpoint.pt file

Once you've added your model to the *ZazuML model zoo* you have to append it to the 
*models.json* file so that *ZazuML* knows to call upon it. 

### *Example key value in model.json*

```
  "retinanet": {
    "task": "detection",
    "model_space": {
      "accuracy_rating": 8,
      "speed_rating": 2,
      "memory_rating": 4
    },
    "hp_search_space": [
      {
        "name": "learning_rate",
        "values": [
          5e-4,
          1e-5,
          5e-5
        ]
      },
      {
        "name": "anchor_scales",
        "values": [
          [1, 1.189207115002721, 1.4142135623730951],
          [1, 1.2599210498948732, 1.5874010519681994],
          [1, 1.5, 2.0]
        ]
      }
    ],
    "training_configs": {
      "epochs": 100,
      "depth": 50,
      "input_size": 608,
      "learning_rate": 1e-5,
      "anchor_scales": [1, 1.2599210498948732, 1.5874010519681994]
    }
  }
```

**hp_search_space** - is for defining hyper-parameters that will over-go optimization 

**training_configs** - is where fixed hyper-parameters are defined

Which parameters will be frozen and which will be optimizable is a design decision 
and will be immutable once the model is pushed to the *ZazuML model zoo*.

**model_space** - is where you define the relative location of your model in a euclidean vector space

**task** - is the defining task of your model, currently you can choose from either 
*classification*, *detection* or *instance segmentation*

The json object key must match the model directory name exactly so that
ZazuML knows what model to call upon, in our example the name of 
both will be ***"retinanet"***.

## Refrences
Thank you to these repositories for their contributions to the ***ZaZoo***

- Yhenon's [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)
- Qqwweee's [keras-yolo3](https://github.com/qqwweee/keras-yolo3)