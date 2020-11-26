from torch import nn
from collections import OrderedDict
from lib.utils.image_list import to_image_list
from lib.networks.processor import Processor
from lib.networks.postprocessor import PostProcessor
from lib.utils.decoding import OutputDecoder
from lib.networks.dla import DLA34

class FADNet(nn.Module):
    def __init__(self, cfg):
        super(FADNet, self).__init__()
        self.backbone = nn.Sequential(OrderedDict([("body", DLA34())]))
        self.processor = Processor(cfg, in_channels=64)

        output_decoder = OutputDecoder(cfg.dim_templates)
        self.post_processor = PostProcessor(
                                output_decoder,
                                cfg.det_thresh,
                                cfg.det_max_obj,
        )

    def forward(self, images, meta_data=None):
        images = to_image_list(images)
        features,base_features = self.backbone(images.tensors)

        x = self.processor(features, base_features)
        result = self.post_processor(x, meta_data)


        return result