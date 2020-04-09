import dtlpy as dl
import logging
from importlib import import_module

logger = logging.getLogger(name=__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Package runner class

    """

    def __init__(self, **kwargs):
        """
        Init package attributes here

        :param kwargs: config params
        :return:
        """
        pass

    def predict_single_item(self, item, progress=None):

        cls = getattr(import_module('.adapter', 'ObjectDetNet.' + 'retinanet'), 'AdapterModel')
        adapter = cls()
        # these lines can be removedpy
        dirname = adapter.predict_item(item, 'checkpoint.pt')

        logger.info('uploaded prediction from ' + dirname)

if __name__ == "__main__":
    """
    Run this main to locally debug your package
    """
    dl.packages.test_local_package()
