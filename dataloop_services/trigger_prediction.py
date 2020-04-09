from push_deploy import push_package, deploy, x
import dtlpy as dl
dl.setenv('prod')
project = dl.projects.get('golden_project')
package = push_package(project)
predict_service = deploy(package, service_name='predict_item')

# item_input = dl.FunctionIO(type='Item', name='item', value={"item_id": ''})
# inputs = [item_input]
# service.execute(function_name='run', execution_input=inputs)

dataset = project.datasets.get('predict_rodent')
dataset_id = dataset.id
filters = dl.Filters()
filters.add(field='datasetId', values=dataset.id)
filters.add(field='metadata.system.annotationStatus', values='completed')

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
