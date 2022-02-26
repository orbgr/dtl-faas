import logging
import os
import time
import dtlpy as dl
from os.path import join as join_path

def main():
    project = dl.projects.get(project_name='Body Parts Detection')
    dataset = project.datasets.get(dataset_name='DB_Customer')
    model_path = join_path(".", "M0.pt")

    logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("deploy_function - Object Detection")
    logger.setLevel(logging.INFO)

    weights_id = '61ddbc0027fe04447d2aa01f'
    # model_path = ""

    # define module
    module = dl.PackageModule(entry_point='main.py',
                              init_inputs=[dl.FunctionIO(name='weights_id', type=dl.PackageInputType.JSON)],
                              functions=[dl.PackageFunction(name='detect',
                                                            description='Detect Body Parts using YoloV5 model',
                                                            inputs=[dl.FunctionIO(name='input_json', type=dl.PackageInputType.JSON)])])

    # define package
    package = project.packages.push(src_path=os.getcwd(), package_name='body-part-detector', modules=[module])

    # define artifacts
    package.artifacts.upload(filepath=model_path, package=package, package_name=package.name)


    # define service
    service = package.deploy(service_name='body-part-detector',
                             init_input=[dl.FunctionIO(name='weights_id',
                                                       type=dl.PackageInputType.JSON,
                                                       value=weights_id)],
                             runtime=dl.KubernetesRuntime(concurrency=1, pod_type=dl.InstanceCatalog.REGULAR_S),
                             cython_runtime=2)

    # configure filter
    # try:
    #     filters = dl.Filters(resource=dl.FiltersResource.ITEM)
    #     filters.add(field="datasetId", values=dataset.id)
    #     filters.add(field='metadata.system.mimetype', values='image*')
    #     filters.add(field='dir', values='/Input')
    #
    #     trigger = service.triggers.create(
    #         name='body-part-detector',
    #         function_name="detect",
    #         resource=dl.TriggerResource.JSON,
    #         actions=[dl.TriggerAction.CREATED],
    #         filters=filters
    #     )
    # except:
    #     logger.info("trigger already exists")

    logger.info("======================== TIME STARTED =======================")
    time.sleep(10)
    service.pause()
    logger.info("======================== SERVICE PAUSED =======================")

if __name__ == '__main__':
    dl.verbose.logging_level = dl.VERBOSE_LOGGING_LEVEL_WARNING
    if dl.token_expired():
        dl.login()
    main()
