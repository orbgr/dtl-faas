# in init:
self.classifier = download.classifier

label = "IS_UNSAFE"
item = "item" # get img item
# item.download
img_path = ""

classifier_res_dict = self.classifier.classify(img_path)
prob_unsafe = classifier_res_dict[img_path]["unsafe"]

if upload:
    label_define = dl.Box(left=item.width - 20,
                          top=item.height - 20,
                          right=item.width,
                          bottom=item.height,
                          label=label)

    builder = item.annotations.builder()
    builder.add(annotation_definition=label_define, model_info={'name': "NudeNet",
                                                                'confidence': prob_unsafe,
                                                                'class_label': label})
    item.annotations.upload(builder)

return prob_unsafe # to json format