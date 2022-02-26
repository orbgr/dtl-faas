import os
import dtlpy as dl
import torch
import json
import logging
from os.path import join as join_path



# TODO: replace filtering to get latest model CORRECTLY! (currently M2 will be taken before M10)
def get_best_model_id(project_name='Body Parts Detection', dataset_name='test') -> str:
    filters = dl.Filters()
    filters.add(field='dir', values="/models")
    filters.sort_by(field=dl.FILTERS_KNOWN_FIELDS_FILENAME)
    project = dl.projects.get(project_name=project_name)
    dataset = project.datasets.get(dataset_name=dataset_name)
    weights_id = dataset.items.get_all_items(filters=filters)[-1].id
    return weights_id

logging.basicConfig(format='[YOAV] -  %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deploy_function - Object Detection")
logger.setLevel(logging.INFO)

# module main.py
class ServiceRunner:
    def __init__(self, weights_id: str = '61ddbc0027fe04447d2aa01f'):
        self.weights_id = weights_id
        self.model_name = "M0.pt"
        self.model_path = join_path(".", self.model_name)
        # hard-coded to Basic NudeNet labels
        self.labels = ("EXPOSED_BELLY", "EXPOSED_BREAST_F",
                       "EXPOSED_BREAST_M", "EXPOSED_BUTTOCKS",
                       "EXPOSED_GENITALIA_F", "EXPOSED_GENITALIA_M")

        self.project = dl.projects.get(project_name='Body Parts Detection')
        self.dataset = self.project.datasets.get(dataset_name='DB_Customer')


        # self.weights_id = get_best_model_id()
        self.package = dl.packages.get(package_name="body-part-detector")
        self.package.artifacts.download(artifact_name=self.model_name, local_path=self.model_path)

    def detect(self, input_json: dl.PackageInputType.JSON = None):
        def build_ant(results, dl_ann_builder):
            dl_ann_builder.add(annotation_definition=dl.Box(left=results["xmin"], bottom=results["ymax"],
                                                            right=results["xmax"], top=results["ymin"],
                                                            label=results["name"]),
                               model_info={'name': f"{self.dataset}-YoloV5",
                                           'confidence': results["confidence"],
                                           'class_label': results["class"]})

        input_json = json.loads(input_json)
        target_img_as_item = input_json["target_img_as_item"]
        logger.info(f"target_img_as_item: {input_json['target_img_as_item']}")
        cfg_labels = input_json["cfg_labels"]
        logger.info(f"cfg_labels: {input_json['cfg_labels']}")
        # get best model
        # best_model_id = get_best_model_id()
        # best_model = dataset.items.get(item_id=best_model_id)
        # best_model.download(local_path='./')
        # best_model_name = best_model.name
        # model_path = f'./{best_model_name}'
        # logger.info(f"[YOAV] - model_path: {model_path}")
        customer_label_list = [label.upper() for label, is_on in cfg_labels.items() if is_on == '1']
        logger.info(f"after customer label list: {customer_label_list}")

        # download image to ./image and get path
        target_img_as_item = self.dataset.items.get(item_id=target_img_as_item)
        target_img_as_item.download(local_path='./image/')
        image_path = f'./image/{target_img_as_item.name}'

        # save current working directory
        root = os.getcwd()

        # load YOLOv5 with best_model weights
        model = torch.hub.load('./', 'custom', path=self.model_path, source='local', force_reload=True)
        # run inference, save results
        res = model(image_path, size=640)
        final_res = res.pandas().xyxy[0]

        logger.info(f"[YOAV] - final_res names: {final_res['name']}")
        final_res = final_res.loc[final_res["name"].isin(customer_label_list)]

        # navigate to root = correct working directory
        os.chdir(root)

        # Upload image
        self.dataset.items.upload(local_path=image_path, remote_path='/Output/')

        # get uploaded image path
        img_path_in_faas = target_img_as_item.filename.split('/')[-1]
        img_in_faas_out = self.dataset.items.get(filepath=f'/Output/{img_path_in_faas}')

        # Upload annotations
        builder = dl.AnnotationCollection(item=img_in_faas_out)
        _ = final_res.apply(lambda x: build_ant(x, builder), axis=1)
        img_in_faas_out.annotations.upload(builder)

        input_json["faas_output"]["yolo_ants"] = final_res
        return final_res


