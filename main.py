from retinanet import AdapterModel

model = AdapterModel()
model.load()
model.preprocess()
model.build()
model.train()