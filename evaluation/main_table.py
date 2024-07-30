import json

paths_to_evaluate = [
    "FD_diff.json",

    "FD_load_diff.json",
    "FD_load_deform.json",
    "FD_load_dino.json",

    "FD_glip.json",
    "FD_gdino.json",

    "FD_BCG_glip.json",
    "FD_BCG_gdino.json",
    "FD_BCG_SOS.json"
]
model_name = [
    'Diffusion-DETR w/o pre-training',

    'Diffusion-DETR',
    'DDETR',
    'DINO',

    'GLIP',
    'Grounding DINO',

    'GLIP-MT',
    'Grounding DINO-MT',
    'FD-SOS',

]
metrics = ['AP50_FD', 'AP_FD', 'mAP50', 'mAP']

map_ = ['anterior teeth FD_precision_50', 'anterior teeth FD_precision_75', 'anterior teeth FD_precision',
        'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP']

for idx, pred in enumerate(paths_to_evaluate):
    with open(pred, 'r') as j:
        pred_contents = json.loads(j.read())
    str_ = ''
    for j, map in enumerate(map_):
        str_ += str(round(pred_contents[map] * 100, 4))
        if j == len(map_) - 1:
            str_ += ' \\\ '
        else:
            str_ += ' & '
    print(model_name[idx], ' & ', str_)
