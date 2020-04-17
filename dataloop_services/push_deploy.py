import dtlpy as dl
import os


def deploy_predict_item(package, model_id, checkpoint_id):
    input_to_init = {'model_id': model_id,
                     'checkpoint_id': checkpoint_id}

    service_obj = package.services.deploy(service_name='predict',
                                          module_name='predict_item_module',
                                          package=package,
                                          runtime={'gpu': False,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          init_input=input_to_init)

    return service_obj


def push_package(project):
    item_input = dl.FunctionIO(type='Item', name='item')
    model_input = dl.FunctionIO(type='Json', name='model_id')
    checkpoint_input = dl.FunctionIO(type='Json', name='checkpoint_id')

    model_inputs = [item_input]
    predict_item_function = dl.PackageFunction(name='predict_single_item', inputs=model_inputs, outputs=[],
                                               description='')
    predict_item_module = dl.PackageModule(entry_point='dataloop_services/prediction_module.py',
                                           name='predict_item_module',
                                           functions=[predict_item_function],
                                           init_inputs=[model_input, checkpoint_input])

    package_obj = project.packages.push(
        package_name='ObDetNet',
        src_path=os.getcwd(),
        modules=[predict_item_module])

    return package_obj
