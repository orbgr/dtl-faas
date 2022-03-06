import json
import os
import platform

import dtlpy as dl
import pandas as pd
import glob
import logging
import subprocess
from os.path import join as join_path
from pyunpack import Archive
# import patoolib

logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pose estimation")
logger.setLevel(logging.INFO)

class ServiceRunner:
    def __init__(self):
        self.package = dl.packages.get(package_name='pose-estimator')

        self.body_dict = {
             0: "Nose",
             1: "Neck",
             2: "RShoulder",
             3: "RElbow",
             4: "RWrist",
             5: "LShoulder",
             6: "LElbow",
             7: "LWrist",
             8: "MidHip",
             9: "RHip",
             10: "RKnee",
             11: "RAnkle",
             12: "LHip",
             13: "LKnee",
             14: "LAnkle",
             15: "REye",
             16: "LEye",
             17: "REar",
             18: "LEar",
             19: "LBigToe",
             20: "LSmallToe",
             21: "LHeel",
             22: "RBigToe",
             23: "RSmallToe",
             24: "RHeel",
             25: "Background"
            }

        dir_name = r"openpose.rar"
        self.openpose_path = join_path(os.getcwd())
        full_path = join_path(self.openpose_path, dir_name)

        logger.info("download artifact start")
        self.package.artifacts.download(artifact_name=dir_name,
                                        local_path=full_path)
        logger.info("download artifact finish")
        logger.info(f'files in dir before extract {glob.glob(join_path(self.openpose_path, "*"))}')

        logger.info(f"platform={platform.platform()}")
        # os.environ['PATH'] += ':' + ""
        # subprocess.run("sudo apt - get install unrar")
        subprocess.run("unzip openpose.zip")

        Archive(full_path).extractall(self.openpose_path)
        # patoolib.extract_archive(full_path, outdir=self.openpose_path)
        logger.info(f'files in dir after extract {glob.glob(join_path(self.openpose_path, "*"))}')

        self.openpose_path = join_path("..", "models", "openpose")
        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')

    # @staticmethod
    def run(self, input: dl.PackageInputType.JSON = None):
        input = json.loads(input)
        item_id = input["item_id"]
        item_img = self.dataset.items.get(item_id=item_id)

        logger.info(f"[YOAV] - item_img: {item_img.name}")

        # item download
        item_img.download(local_path=join_path(self.openpose_path, "input"))
        # cd

        logger.info(f"[YOAV] before cd- {os.getcwd()}")
        os.chdir(self.openpose_path)

        # run
        shell_output = subprocess.run(  "./bin/OpenPoseDemo.exe "
                                        "--image_dir ./input/ "
                                        "-display 0 "
                                        "--write_json ./output/ "
                                        "--write_images ./output_images/ ",
                                         capture_output=True
                                     ).stdout

        logger.info(f"[YOAV] shell_output = {shell_output}")
        logger.info(f"[YOAV] after cd- {os.getcwd()}")

        paths = []
        # todo: multiple images in the same session - clean input & output dirs
        for path, subdirs, files in os.walk(join_path(os.getcwd(), "openpose", "input", "*")):
            for name in files:
                paths.append(join_path(path, name))
        for path, subdirs, files in os.walk(join_path(os.getcwd(), "openpose", "output", "*")):
            for name in files:
                paths.append(join_path(path, name))
        logger.info(f"[YOAV] - { paths }")

        # use the json
        key_points_df = pd.DataFrame()

        for json_path in glob.glob("output/*"):
            try:
                key_points = pd.json_normalize(pd.read_json(json_path)["people"])["pose_keypoints_2d"]
                key_points.name = "cordi"
                key_points.index = len(key_points.index) * [json_path]

                key_points_df = pd.concat([key_points_df, key_points], axis=0)
            except KeyError:
                logger.error(f"[YOAV] - Missing pose ants in {json_path}")
                continue

        logger.info(f"[YOAV] - JSON: {key_points_df}")

        def to_ants(x):
            res = pd.Series(dtype=object)

            for i, arr in enumerate(list(zip(x[0::3], x[1::3], x[2::3]))):
                res[self.body_dict[i]] = arr

            return res

        key_points_df = key_points_df.apply(lambda x: to_ants(x[0]), axis=1)

        def build_ant(x, builder, parent_annotation):
            cor = x[0]
            # 0's + bakround
            if cor[0] == 0 and cor[1] == 0 and cor[2] == 0:
                return
            if x.name == "Background":
                return

            builder.add(annotation_definition=dl.Point(x=cor[0], y=cor[1], label=x.name),
                        parent_id=parent_annotation.id,
                        model_info={'name': "pose",
                                    'confidence': cor[2],
                                    'class_label': x.name})
            return

        def upload_pose(item, img_in_faas_out, item_keypoints, recipe, template_id, duplicates="skip"):
            # img_name = item_keypoints.name.split("\\")[-1].split(".")[0].replace("_keypoints", "")
            # item = items_dict[img_name]

            annotation_id_old = [ant.id for ant in item.annotations.list() if ant.type == "pose"]

            if len(annotation_id_old) > 0:

                if duplicates == "delete":
                    annotation = item.annotations.get(annotation_id=annotation_id_old[0])
                    annotation.delete()

                if duplicates == "skip":
                    return

            parent_annotation = item.annotations.upload(
                dl.Annotation.new(annotation_definition=dl.Pose(label='pose',
                                                                template_id=template_id,
                                                                # instance_id is optional
                                                                instance_id=None)))[0]

            builder = item.annotations.builder(item=img_in_faas_out)
            _ = item_keypoints.to_frame().apply(lambda x: build_ant(x, builder, parent_annotation), axis=1)
            builder.upload()


        # Upload image
        self.dataset.items.upload(local_path=f"./input/{item_img.name}", remote_path='/FaaS_out/')

        # get uploaded image path
        img_path_in_faas = item_img.filename.split('/')[-1]
        img_in_faas_out = self.dataset.items.get(filepath=f'/FaaS_out/{img_path_in_faas}')


        recipe = self.dataset.recipes.list()[0]
        template_id = recipe.get_annotation_template_id(template_name="pose")

        _ = key_points_df.apply(lambda x: upload_pose(item_img, img_in_faas_out, x, recipe, template_id), axis=1)

        os.chdir("..")
        return 0

