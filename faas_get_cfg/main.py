import os
import dtlpy as dl
import json
import logging
from configparser import ConfigParser
import time 
import json
from os.path import join as join_path

logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - get cfg")
logger.setLevel(logging.INFO)


# module main.py
class ServiceRunner:
    def __init__(self):
        self.faas_dict = { "1":["object-detection"],
                           "2":["object-detection"], # , "pose-estimation"
                           "3":["object-detection", "classifier"] # , "pose-estimation"
                          }
        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')

        self.output_type_to_is_upload = dict(hide="1", proba="0", both="1")

    def _get_cfg(self):
        logger.info(f"[YOAV] - Getting CFG")

        filters = dl.Filters(field='datasetId', values=self.dataset.id)
        filters.add(field='dir', values='/Config')
        filters.add(field='name', values=f'cfg_{self.customer_id}.ini')

        try:
            cfg_file = self.dataset.items.list(filters=filters)[0][0]
        except:
            logger.error("[YOAV ERROR] no cfg for customer:", self.customer_id)

        cfg_file.download(f"./cfg_files/cfg_{self.customer_id}.ini")
        # read
        config = ConfigParser()
        config.read(f'cfg_files/cfg_{self.customer_id}.ini')
        logger.info(f"[YOAV] { config } ")
        logger.info(f'[YOAV] { config["Models"]["accuracy"] } ')

        return config

    def _exe_object_detection(self, faas_name):
        input = dict(item_id=self.item.id,
                     cfg_labels=self.cfg._sections["Labels"],
                     upload=self.output_type_to_is_upload[self.cfg._sections["Models"]["output_type"]])

        logger.info(f"item_id before faas {self.item.id}")
        logger.info(f"cfg_labels before faas {self.cfg._sections['Labels']}")

        input = json.dumps(input, indent=4)

        func_a = dl.FunctionIO(name='input',
                               value=input,
                               type=dl.PackageInputType.JSON)

        execution = dl.services.get(service_name=faas_name)

        return dict(execution=execution, execution_input=func_a, function_name="detect")

    def _exe_recognition_engine(self, faas_name):
        input = dict(item_id=self.item.id,
                      cfg=self.cfg._sections,
                      output_models=self.recognition_engine_input)
        logger.info(f"item_id before faas {self.item.id}")

        input = json.dumps(input, indent=4)

        func_a = dl.FunctionIO(name='input',
                               value=input,
                               type=dl.PackageInputType.JSON)


        execution = dl.services.get(service_name=faas_name)

        return dict(execution=execution, execution_input=func_a, function_name="run")

    def _exe_classifier(self, faas_name):
        input = dict(item_id=self.item.id,
                          upload=self.output_type_to_is_upload[self.cfg._sections["Models"]["output_type"]])

        logger.info(f"item_id before faas {self.item.id}")

        input = json.dumps(input, indent=4)

        func_a = dl.FunctionIO(name='input',
                               value=input,
                               type=dl.PackageInputType.JSON)

        execution = dl.services.get(service_name=faas_name)
        return dict(execution=execution, execution_input=func_a, function_name="run")

    def _execute_faas(self, faas_name: dl.PackageInputType.JSON=None):
        logger.info(f"FaaS GET_CFG before running : {faas_name}")

        if faas_name == "object-detection":
            return self._exe_object_detection(faas_name)

        elif faas_name == "recognition-engine":
            return self._exe_recognition_engine(faas_name)

        elif faas_name == "classifier":
            return self._exe_classifier(faas_name)

        else:
            logger.error(f"there's no faas name {faas_name}")

    def run(self, item_img: dl.Item=None):

        self.item = item_img
        self.customer_id = item_img.name.split(".")[0].split("_")[-1]
        # self.customer_id = "abc123"
        root = os.getcwd()
        logger.info(f"customer_id: {self.customer_id}")
        
        # get_cfg from dtl
        # self.cfg = self._get_cfg()
        logger.info(f"Getting CFG")
        filters = dl.Filters(field='datasetId', values=self.dataset.id)
        filters.add(field='dir', values='/Config')
        filters.add(field='name', values=f'cfg_{self.customer_id}.ini')

        try:
            cfg_file = self.dataset.items.list(filters=filters)[0][0]
        except:
            logger.error("no cfg for customer:", self.customer_id)

        cfg_file.download(f"./cfg_files/cfg_{self.customer_id}.ini")
        # read
        self.cfg = ConfigParser()
        self.cfg.read(f'cfg_files/cfg_{self.customer_id}.ini')
        logger.info(f'cfg models accuracy { self.cfg["Models"]["accuracy"] } ')

        # Upload image
        item_img.download(local_path='./image/')
        image_path = f'./image/{item_img.name}'
        self.dataset.items.upload(local_path=image_path, remote_path='/Output/')
        img_name = item_img.filename.split('/')[-1]
        self.item = self.dataset.items.get(filepath=f'/Output/{img_name}')

        execution_dict = {}
        # init
        faas_to_exe = self.faas_dict[self.cfg["Models"]["accuracy"]]
        logger.info(f"FaaS to Execute list: {faas_to_exe}")
        for i, faas_name in enumerate(faas_to_exe):
            execution_dict[faas_name] = self._execute_faas(faas_name)

        # execute
        for i, faas_name in enumerate(faas_to_exe):
            execution_input = execution_dict[faas_name]["execution_input"]
            function_name = execution_dict[faas_name]["function_name"]

            logger.info(f"from execution execute loop - starting execute {i}, {faas_name} with func_name={function_name} & exe_input={execution_input}")

            execution_dict[faas_name] = execution_dict[faas_name]["execution"].execute(execution_input=execution_input,
                                                                                       function_name=function_name,
                                                                                       project_id=self.project.id)

        # wait
        for faas_name, exe in execution_dict.items():
            _ = exe.wait()

        # output usage
        output_dict = {}
        for faas_name, exe in execution_dict.items():
            logger.info(f"from execution output loop - starting execute {i}, {faas_name}")
            try:
                output = exe.get_latest_status()["output"]
                output_dict[faas_name] = output
                logger.info(f"Output for {faas_name}: {output}")

            except Exception as e:
                logger.info("Output for {faas_name}: exe.get_latest_status()")
                logger.error(f"I'm here because of exception: {e}")
                continue

        self.recognition_engine_input = output_dict
        self._execute_faas("recognition-engine")
