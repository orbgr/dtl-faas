import logging
import os
import time
import dtlpy as dl
from datetime import datetime, timedelta


def main():
    project = dl.projects.get(project_name='Body Parts Detection')
    dataset = project.datasets.get(dataset_name='DB_Customer')
    faas_name = "get-config"

    logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("deploy function - get cfg")
    logger.setLevel(logging.INFO)

    # define module
    module = dl.PackageModule(entry_point='main.py',
                              init_inputs=[],
                              functions=[dl.PackageFunction(name='run',
                                                            description='Running FaaS using Customer config',
                                                            inputs=[dl.FunctionIO(name='item_img', type=dl.PackageInputType.ITEM)]
                                                            )])

    # define package
    package = project.packages.push(src_path=os.getcwd(), package_name=faas_name, modules=[module])



    # define service
    service = package.deploy(service_name=faas_name,
                             init_input=[],
                             runtime=dl.KubernetesRuntime(concurrency=1, pod_type=dl.InstanceCatalog.REGULAR_S),
                             cython_runtime=2)

    # configure filter
    try:
        filters = dl.Filters(resource=dl.FiltersResource.ITEM)
        filters.add(field='metadata.system.mimetype', values='image*')
        filters.add(field='dir', values='/Input')

        trigger = service.triggers.create(
            name=faas_name,
            project_id=project.id,
            function_name='run',
            resource=dl.TriggerResource.ITEM,
            actions=dl.TriggerAction.CREATED,
            filters=filters
        )
    except:
        logger.error("problem uploading trigger")

    # logger.info("======================== TIME STARTED =======================")
    # time.sleep(5)
    # service.pause()
    # logger.info("======================== SERVICE PAUSED =======================")

if __name__ == '__main__':
    dl.verbose.logging_level = dl.VERBOSE_LOGGING_LEVEL_WARNING
    if dl.token_expired():
        dl.login()
    main()