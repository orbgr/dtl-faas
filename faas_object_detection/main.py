import os
import dtlpy as dl
import torch
import json
import logging
from os.path import join as join_path



def get_last_model_item(project_name='Body Parts Detection', dataset_name='Content_filter') -> dl.Item:
    filters = dl.Filters()
    filters.add(field='dir', values="/Models")
    project = dl.projects.get(project_name=project_name)
    dataset = project.datasets.get(dataset_name=dataset_name)

    model_items = dataset.items.get_all_items(filters=filters)
    model_names = {item.name: item for item in model_items if item.name.startswith("M")}
    max_iter = max([int(model_name.replace("M", "").replace(".pt", "")) for model_name in model_names.keys()])

    # todo: hardcoded :(
    max_iter = 0

    return model_names[f"M{max_iter}.pt"]

logging.basicConfig(format='[YOAV] -  %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("object Detection")
logger.setLevel(logging.INFO)

# todo: labels not hard coded, need to use M1 new recipe
class ServiceRunner:
    def __init__(self):
        self.item_model = get_last_model_item()
        self.model_path = join_path(".", self.item_model.name)
        self.item_model.download(self.model_path)
        # hard-coded to Basic NudeNet labels
        self.labels = ("EXPOSED_BELLY", "EXPOSED_BREAST_F",
                       "EXPOSED_BREAST_M", "EXPOSED_BUTTOCKS",
                       "EXPOSED_GENITALIA_F", "EXPOSED_GENITALIA_M")

        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')

        # self.package = dl.packages.get(package_name="object-detection")
        # self.package.artifacts.download(artifact_name=self.model_name, local_path=self.model_path)

    def detect(self, input: dl.PackageInputType.JSON = None):
        def build_ant(results, dl_ann_builder):
            dl_ann_builder.add(annotation_definition=dl.Box(left=results["xmin"], bottom=results["ymax"],
                                                            right=results["xmax"], top=results["ymin"],
                                                            label=results["name"]),
                               model_info={'name': f"{self.dataset}-YoloV5",
                                           'confidence': results["confidence"],
                                           'class_label': results["class"]})

        # input
        input = json.loads(input)
        item_id = input["item_id"]
        cfg_labels = input["cfg_labels"]
        logger.info(f"cfg_labels: {input['cfg_labels']} |  item_id: {input['item_id']}")

        customer_label_list = [label.upper() for label, is_on in cfg_labels.items() if is_on == '1']
        logger.info(f"after customer label list: {customer_label_list}")

        # target item
        item_img = self.dataset.items.get(item_id=item_id)
        item_img.download(local_path='./image/')
        image_path = f'./image/{item_img.name}'

        # save current working directory
        root = os.getcwd()

        # load YOLOv5 with best_model weights
        model = torch.hub.load('./', 'custom', path=self.model_path, source='local', force_reload=True)
        # run inference, save results
        detection_output = model(image_path, size=640)
        detection_output = detection_output.pandas().xyxy[0]
        logger.info(f"detection_output before: {detection_output}")
        detection_output = detection_output.loc[detection_output["name"].isin(customer_label_list)]
        logger.info(f"detection_output after: {detection_output}")

        # navigate to root = correct working directory
        os.chdir(root)

        if input["upload"] == "1":
            # Upload annotations
            builder = dl.AnnotationCollection(item=item_img)
            _ = detection_output.apply(lambda x: build_ant(x, builder), axis=1)
            item_img.annotations.upload(builder)

        return json.dumps(detection_output.to_dict(), indent=4)

