import logging
import os
import dtlpy as dl


def main():
    project = dl.projects.get(project_name='Body Parts Detection')
    dataset = project.datasets.get(dataset_name='DB_Customer')
    faas_name = "recognition-engine"

    logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("deploy function - rec engine")
    logger.setLevel(logging.INFO)

    # define module
    module = dl.PackageModule(entry_point='main.py',
                              init_inputs=[],
                              functions=[dl.PackageFunction(name='run',
                                                            description='Merging all models output to an output',
                                                            inputs=[dl.FunctionIO(name='input', type=dl.PackageInputType.JSON)]
                                                            )])

    # define package
    package = project.packages.push(src_path=os.getcwd(), package_name=faas_name, modules=[module])



    # define service
    service = package.deploy(service_name=faas_name,
                             init_input=[],
                             runtime=dl.KubernetesRuntime(concurrency=1, pod_type=dl.InstanceCatalog.REGULAR_S),
                             cython_runtime=2)


if __name__ == '__main__':
    dl.verbose.logging_level = dl.VERBOSE_LOGGING_LEVEL_WARNING
    if dl.token_expired():
        dl.login()
    main()