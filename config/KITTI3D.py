from easydict import EasyDict

def get_configs():
    cfg=EasyDict()

    cfg.data_root = './data/object_detection/testing'
    cfg.pretrained_model="./checkpoint/final.pth"
    cfg.output_dir = "./output"

    cfg.backbone_stride = 4
    cfg.input_width = 1280
    cfg.input_height = 384
    cfg.det_thresh=0.25
    cfg.det_max_obj=50

    cfg.det_classes = ("Car", "Cyclist", "Pedestrian")
    cfg.input_mean=[0.485, 0.456, 0.406]
    cfg.input_std=[0.229, 0.224, 0.225]
    cfg.dim_templates = ((3.88, 1.53, 1.63),
                         (1.78, 1.70, 0.58),
                         (0.88, 1.73, 0.67))

    return cfg


