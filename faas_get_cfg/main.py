import os
import dtlpy as dl
import torch
import json
import logging
from configparser import ConfigParser
import time 
import threading
import json

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - get cfg")
logger.setLevel(logging.INFO)


# module main.py
class ServiceRunner:
    def __init__(self):
        self.faas_dict = { "1":["body-part-detector"],
                           "2":["body-part-detector"],
                           "3":["body-part-detector"]
                          }
        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')

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

    def _exe_object_detection(self, faas_output):
        input_json = dict(target_img_as_item=self.item.id,
                          faas_output=faas_output,
                          cfg_labels=self.cfg._sections["Labels"])

        logger.info(f"target_img_as_item {self.item.id}")
        logger.info(f"cfg_labels {self.cfg._sections['Labels']}")

        input_json = json.dumps(input_json, indent=4)

        func_a = dl.FunctionIO(name='input_json',
                               value=input_json,
                               type=dl.PackageInputType.JSON)

        execution = dl.services.get(service_name='body-part-detector').execute(execution_input=func_a,
                                                                               function_name="detect",
                    #                                                            resource="dl.PackageInputType.ITEM",
                    #                                                            item_id=item.id,
                                                                               project_id=self.project.id)
        time.sleep(10)
        execution.logs()
        return execution

    def _exe_recognition_engine(self):
        input_json = dict(input_img_item=self.item.id,
                          cfg=self.cfg._sections,
                          output_models=self.recognition_engine_input)

        input_json = json.dumps(input_json, indent=4)

        func_a = dl.FunctionIO(name='input_json',
                               value=input_json,
                               type=dl.PackageInputType.JSON)


        execution = dl.services.get(service_name='recognition-engine').execute(execution_input=func_a,
                                                                               function_name="run",
                    #                                                            resource="dl.PackageInputType.ITEM",
                    #                                                            item_id=item.id,
                                                                               project_id=self.project.id)

        # time.sleep(10)
        # execution.logs()
        # return execution

    def _execute_faas(self, faas_name: dl.PackageInputType.JSON=None, faas_output=None):
        logger.info(f"[YOAV] - FaaS GET_CFG before running : {faas_name}")

        if faas_name == "body-part-detector":
            faas_output = self._exe_object_detection(faas_output)
            return faas_output

        elif faas_name == "recognition-engine":
            return self._exe_recognition_engine()

    def run(self, input_img_item: dl.Item=None):

        # self.customer_id = input_img_item.name.split(".")[0].split("_")[-1]
        self.customer_id = "abc123"
        root = os.getcwd()
        logger.info(f"[YOAV] - customer_id: {self.customer_id}")
        
        # get_cfg from dtl
        # self.cfg = self._get_cfg()
        logger.info(f"[YOAV] - Getting CFG")

        filters = dl.Filters(field='datasetId', values=self.dataset.id)
        filters.add(field='dir', values='/Config')
        filters.add(field='name', values=f'cfg_{self.customer_id}.ini')

        try:
            cfg_file = self.dataset.items.list(filters=filters)[0][0]
        except:
            logger.info("[YOAV ERROR] no cfg for customer:", self.customer_id)

        cfg_file.download(f"./cfg_files/cfg_{self.customer_id}.ini")
        # read
        self.cfg = ConfigParser()
        self.cfg.read(f'cfg_files/cfg_{self.customer_id}.ini')
        logger.info(f'[YOAV] cfg models accuracy { self.cfg["Models"]["accuracy"] } ')
    
        # upload image to dtl input
        self.item = input_img_item
#         input_img_item.download(local_path='./image/')
#         image_path = f'./image/{input_img_item.name}'
        
        # call object detection
        threads = {}
        output_dict = {}
        for i, faas_name in enumerate(self.faas_dict[self.cfg["Models"]["accuracy"]]):
            output_dict[faas_name] = {}
            # threads[i] = threading.Thread(target=self._execute_faas, args=(faas_name, output_dict[faas_name]))
            res = self._execute_faas(faas_name, output_dict[faas_name])

        # for i, t in threads.items():
        #     t.start()
        #
        # for i, t in threads.items():
        #     t.join()
        logger.info(f"[YOAV] res-output: {res.output}")
        self.recognition_engine_input = output_dict

        self._execute_faas("recognition-engine")
