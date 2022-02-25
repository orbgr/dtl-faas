import os
import dtlpy as dl
import torch
import json
import logging
import time

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - rec engine")
logger.setLevel(logging.INFO)

# module main.py
class ServiceRunner:
    def __init__(self):
        self.gmb_model_path = "path"


    @staticmethod
    def run(self, output_dict: dl.PackageInputType.JSON=None):
        cfg = output_dict["cfg"]
        input_img_item = output_dict["input_img_item"]
        output_models = output_dict["output_models"]

        root = os.getcwd()
        logger.info("[Yoav] starting recogintion engine run fucntion")


        # get from cfg customer output type
        if cfg["Models"]["output_type"] == "hide":
            # build black box using output_dict["yolo_ants"]
            # upload image to Output
            return

        if cfg["Models"]["output_type"] != "proba":
            logger.info("enter incorrectly output type, will output probability as default behaivour")

        # predict with gmb model
        #         input_img_item.download(local_path='./image/')
        #         image_path = f'./image/{input_img_item.name}'
        # upload image to Output
        return

