import os
import dtlpy as dl
import torch
import json
import logging
import time
from os.path import join as join_path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.utils import save_image
import glob

logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - recogintion engine")
logger.setLevel(logging.INFO)


# module main.py
class ServiceRunner:
    def __init__(self):
        self.gmb_model_path = "path"
        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')

    # @staticmethod
    def run(self, input: dl.PackageInputType.JSON=None):
        input = json.loads(input)

        output_models = input["output_models"]
        cfg = input["cfg"]
        item_id = input["item_id"]

        item_img = self.dataset.items.get(item_id=item_id)
        item_img.download(local_path='./image/')
        image_path = f'./image/{item_img.name}'

        logger.info(f"starting recogintion engine run fucntion with input: cfg={cfg}, item_id={item_id}, output_models={output_models}")

        path_json = f"{item_img.name.split('.')[0]}.json"
        with open(path_json, 'w') as f:
            json.dump(output_models, f)
        self.dataset.items.upload(local_path=path_json, remote_path="/Json/")

        def black_ant(x):
            if x["name"] not in customer_unsafe_list:
                return
            else:
                img_tensor[:,
                           int(x["ymin"]):int(x["ymax"]),
                           int(x["xmin"]):int(x["xmax"])] = 0  # [color, y (300 to 0), x(0 to 400)]
            return


        # change to black
        if cfg["Models"]["output_type"] in ["hide", "both"]:
            picture = Image.open(image_path)
            convert_tensor = transforms.ToTensor()
            img_tensor = convert_tensor(picture)

            customer_unsafe_list = [label.upper() for label, is_on in cfg["Labels"].items() if is_on == '1']
            yolo_res_df = pd.DataFrame(json.loads(output_models['object-detection']))
            logger.info(f"yolo_res_df = {yolo_res_df}")
            _ = yolo_res_df.apply(lambda x: black_ant(x), axis=1)

            black_img_path = join_path("image", f"black_{item_img.name}")
            logger.info(f"img_tensor after = {img_tensor}")

            save_image(img_tensor, black_img_path)
            logger.info(f'images = {glob.glob(join_path("image", "*"))}')

            # upload image
            self.dataset.items.upload(local_path=black_img_path, remote_path='/Output/')

        elif cfg["Models"]["output_type"] in ["proba"]:
            logger.info("TODO - future classification model")

        else:
            logger.error("enter incorrectly output type, will output probability as default behaivour")

        # predict with gmb model
        #         input_img_item.download(local_path='./image/')
        #         image_path = f'./image/{input_img_item.name}'
        # upload image to Output
        return

