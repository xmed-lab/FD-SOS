import copy
from mmdet.evaluation.metrics.coco_metric import CocoMetric
import json
import os
import numpy as np



root_dir = r'./predictions'
preds_folders_to_evaluate = [
    f"FD_diff",

    f"FD_load_deform",
    f"FD_load_dino",
    f"FD_load_diff",

    f"FD_glip",
    f"FD_gdino",

    f"FD_BCG_glip",
    f"FD_BCG_gdino",
    f"FD_BCG_SOS",

]

for _pred_dir in preds_folders_to_evaluate:

    pred_path = os.path.join(root_dir, _pred_dir)

    if os.path.exists(os.path.join('evaluation', pred_path.split('/')[-1] + '.json')):
        print(pred_path, ' evaluated')
        continue

    if 'BCG' in pred_path:
        class_name = (
            "posterior teeth",
            "anterior teeth",
            "anterior teeth No FD",  # would it work?
            "anterior teeth FD",  # would it work?
        )
        test_json = f'40_FD_BCG_test.json'
    else:

        class_name = (
            "anterior teeth No FD",
            "anterior teeth FD",
        )
        test_json = f'40_FD_test.json'

    coco = CocoMetric(
        ann_file=f'data/v1/{test_json}'
        , classwise=True)

    coco.cat_ids = coco._coco_api.get_cat_ids(cat_names=list(class_name))
    coco.img_ids = coco._coco_api.get_img_ids()

    results = []

    for img_idx in range(len(coco.img_ids)):
        image_meta_info = coco._coco_api.loadImgs(coco.img_ids[img_idx])
        image_file = image_meta_info[0]['file_name']
        result_1 = dict()

        with open(os.path.join(pred_path, 'preds', image_file.split('.')[0]) + '.json', 'r') as j:
            pred_contents = json.loads(j.read())

        result_1['img_id'] = coco.img_ids[img_idx]
        result_1['bboxes'] = copy.deepcopy(np.array(pred_contents['bboxes']))
        result_1['scores'] = copy.deepcopy(np.array(pred_contents['scores']))
        result_1['labels'] = copy.deepcopy(np.array(pred_contents['labels']))
        gt = dict()

        coco.results.append((None, result_1))
    _metrics = coco.compute_metrics(coco.results)

    print(_metrics)

    json_object = json.dumps(_metrics, indent=4)
    with open(os.path.join('evaluation', pred_path.split('/')[-1] + '.json'), "w") as outfile:
        outfile.write(json_object)
