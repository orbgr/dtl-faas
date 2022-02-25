import os
import dtlpy as dl
import torch
import json
import logging
import time

logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - classifier")
logger.setLevel(logging.INFO)

# module main.py
class ServiceRunner:
    def __init__(self):
        # self.package = dl.packages.get(package_name='classifier')
        #
        # self.package.artifacts.download(artifact_name='',
        #                                 local_path='')
        logger.info("start classifer...")

    @staticmethod
    def run(input: dl.PackageInputType.JSON=None):
        logger.info("start run...")
        logger.info(f"input {input} type {type(input)}")

        input = json.loads(input)
        a = input["a"]
        b = input["b"]
        time.sleep(10)
        input_json = dict(a=a,
                          b=b,
                          c=3)
        input_json = json.dumps(input_json, indent=4)

        return input_json
