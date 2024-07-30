_base_ = 'glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'

data_root = 'data/v1/'
class_name = ("anterior teeth No FD",
              "anterior teeth FD")  # ,"2nd Molar","2nd Premolar","3rd Molar","Canine","Central Incisor","Lateral Incisor")

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                                             (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])

model = dict(bbox_head=dict(early_fuse=True, use_checkpoint=True, num_classes=num_classes))

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='90_FD_train.json',
        data_prefix=dict(img='images_all/')))

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='20_FD_val.json',
        data_prefix=dict(img='images_all/')))
# test_dataloader = dict(
#     dataset=dict(
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='test/_annotations_two.coco.json',
#         data_prefix=dict(img='test/')))

val_evaluator = dict(ann_file=data_root + '20_FD_val.json')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
# test_evaluator = dict(ann_file=data_root + 'test/_annotations_two.coco.json')

max_epoch = 50

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))

train_cfg = dict(max_epochs=max_epoch, val_interval=1)

vis_backends = [
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'mmdetection_250_FD',
             'group': 'glip_OOD_baseline'
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
            'language_model': dict(lr_mult=0),
        }))

###Base batch_size 2x8
auto_scale_lr = dict(base_batch_size=16)
