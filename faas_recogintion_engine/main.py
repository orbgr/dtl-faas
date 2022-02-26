import os
import dtlpy as dl
import torch
import json
import logging
import time

logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - rec engine")
logger.setLevel(logging.INFO)

# module main.py
class ServiceRunner:
    def __init__(self):
        self.gmb_model_path = "path"


    # @staticmethod
    def run(self, input: dl.PackageInputType.JSON=None):
        input = json.loads(input)

        cfg = input["cfg"]
        item_id = input["item_id"]
        output_models = input["output_models"]

        root = os.getcwd()
        logger.info(f"starting recogintion engine run fucntion with input: cfg={cfg}, item_id={item_id}, output_models={output_models}")


        # get from cfg customer output type
        # if cfg["Models"]["output_type"] in ["hide", "both"]:
        #     # build black box using output_dict["yolo_ants"]
        #     # upload image to Output
        #     return
        #
        # if cfg["Models"]["output_type"] != "proba":
        #     logger.info("enter incorrectly output type, will output probability as default behaivour")

        # predict with gmb model
        #         input_img_item.download(local_path='./image/')
        #         image_path = f'./image/{input_img_item.name}'
        # upload image to Output
        return

