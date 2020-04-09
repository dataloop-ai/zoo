from .push_deploy import push_package, deploy, x
import dtlpy as dl
dl.setenv('prod')

predict_service = dl.services.get('predict_single_item')

project = dl.projects.get('golden_project')
dataset = project.datasets.get('predict_rodent')
dataset_id = dataset.id

#TODO: where does dataset id go?
trigger = predict_service.triggers.create(
    service_id=predict_service.id,
    resource=dl.TriggerResource.ITEM,
    actions=dl.TriggerAction.CREATED,
    name='predict_on_new',
    filters={
        "metadata": {
          "system": {
            "mimetype": {
              "$eq": "image/*"
            }
          }
        }
      },
    execution_mode=dl.TriggerExecutionMode.ONCE,
    function_name='predict_single_item',
    project_id=project.id
)


item = dataset.items.list().items[0]
