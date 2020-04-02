from retinanet import AdapterModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--predict", action='store_true', default=False)
parser.add_argument("--predict_single", action='store_true', default=False)
args = parser.parse_args()

model = AdapterModel()
if args.train:
    model.load()
    model.preprocess()
    model.build()
    model.train()
if args.predict:
    model.predict()
if args.predict_single:
    model.predict_single_image(image_path='/home/noam/0120122798.jpg')