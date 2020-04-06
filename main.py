from retinanet import AdapterModel
import argparse
import dtlpy as dl

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--predict", action='store_true', default=False)
parser.add_argument("--predict_single", action='store_true', default=False)
parser.add_argument("--predict_item", action='store_true', default=False)
args = parser.parse_args()

model = AdapterModel()
if args.train:
    model.load('example_checkpoint.pt')
    model.preprocess()
    model.build()
    model.train()
    model.get_checkpoint()
    model.save()
if args.predict:
    model.predict()
if args.predict_single:
    model.predict_single_image(image_path='/home/noam/0120122798.jpg')
if args.predict_item:
    project = dl.projects.get('buffs_project')
    dataset = project.datasets.get('tiny_mice_p')
    item = dataset.items.get('/items/253597.jpg')
    # filters = dl.Filters(field='filename', values='/items/253*')
    # pages = dataset.items.list(filters=filters)
    # items = [item for page in pages for item in page]
    items = [item]
    model.predict_items(items, 'checkpoint.pt')
