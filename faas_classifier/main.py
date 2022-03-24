import dtlpy as dl
import json
import logging
from nudenet import NudeClassifier
from os.path import join as join_path
import os

logging.basicConfig(format='[YOAV] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy function - classifier")
logger.setLevel(logging.INFO)

# module main.py
class ServiceRunner:
    def __init__(self):
        logger.info("start init classifer...")
        file_name = "nudenet_classifier.onnx"
        self.label = "IS_UNSAFE"
        self.package = dl.packages.get(package_name='classifier')

        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')
        dataset_main = self.project.datasets.get(dataset_name='Content_filter')
        item_classifier = dataset_main.items.get(filepath=f"/Models/{file_name}")

        home = os.path.expanduser("~")
        logger.info(f"path of 'home' {home}")
        _ = item_classifier.download(local_path=join_path(home, ".NudeNet", "classifier_model.onnx"))

        # self.package.artifacts.download(artifact_name=file_name,
        #                                 local_path=join_path("..", ".NudeNet", "classifier_model.onnx")
        #                                 # local_path=join_path("model", file_name)
        #                                 )
        self.classifier = NudeClassifier()

    # @staticmethod
    def run(self, input: dl.PackageInputType.JSON=None):
        logger.info("start running clasiffier...")

        input = json.loads(input)
        logger.info(f"input {input}")


        # download image to ./image and get path
        item_img = self.dataset.items.get(item_id=input["item_id"])
        item_img.download(local_path='./image/')
        img_path = f'./image/{item_img.name}'

        classifier_res_dict = self.classifier.classify(img_path)
        prob_unsafe = classifier_res_dict[img_path]["unsafe"]
        logger.info(f"res of classifier {classifier_res_dict} with proba {prob_unsafe}")

        if input["upload"] == "1":
            # todo: width & height are None :(
            logger.info(f"item_img width={item_img.width} height={item_img.height}")
            try:
                label_define = dl.Box(left=item_img.width - 20,
                                      top=item_img.height - 20,
                                      right=item_img.width,
                                      bottom=item_img.height,
                                      label=self.label)
            except:
                label_define = dl.Box(left=220 - 20,
                                      top=320 - 20,
                                      right=220,
                                      bottom=320,
                                      label=self.label)

            builder = item_img.annotations.builder()
            builder.add(annotation_definition=label_define, model_info={'name': "NudeNet",
                                                                        'confidence': prob_unsafe,
                                                                        'class_label': self.label})
            item_img.annotations.upload(builder)


        return json.dumps(dict(prob_unsafe=prob_unsafe), indent=4)

