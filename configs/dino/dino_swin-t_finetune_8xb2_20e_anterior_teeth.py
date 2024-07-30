_base_ = 'dino_swin-t_finetune_16xb2_1x_coco.py'

# load_from = '/home/mkfmelbatel/mmdetection/tooth_robo_wdb/best_coco_bbox_mAP_epoch_20.pth'  # noqa
data_root = 'data/teeth_robo/'
class_name = ("posterior teeth",
              "anterior teeth")  # ,"2nd Molar","2nd Premolar","3rd Molar","Canine","Central Incisor","Lateral Incisor")

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                                             (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations_two.coco.json',
        data_prefix=dict(img='train/')))

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/_annotations_two.coco.json',
        data_prefix=dict(img='valid/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test/_annotations_two.coco.json',
        data_prefix=dict(img='test/')))

val_evaluator = dict(ann_file=data_root + 'valid/_annotations_two.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations_two.coco.json')

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))

train_cfg = dict(max_epochs=max_epoch, val_interval=1)

vis_backends = [
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'mmdetection',
             'group': 'dino_anterior_swin-t_fine_tune_teeth_robo'
         })
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

###Base batch_size 2x8
auto_scale_lr = dict(base_batch_size=16)
