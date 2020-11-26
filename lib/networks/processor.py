import torch
from torch import nn
from torch.nn import functional as F

from lib.utils.heatmap_op import sigmoid_hm
from lib.layers.GN import group_norm
from lib.networks.convGRU import ConvGRU

def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels[:4])

    return slice(s, e, 1)

class Processor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Processor, self).__init__()

        self.convGRU = ConvGRU(input_channels=in_channels,
                               hidden_channels=[64, ],
                               kernel_size=3, step=4,
                               effective_step=[0, 1, 2, 3]).cuda()

        classes = len(cfg.det_classes)
        regression_channels = (1, 2, 3, 2, 2, 2)
        head_conv = 256

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,),
            group_norm(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      padding=0,)
        )

        self.center2d_wh_head = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,padding=1, ),
            group_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,4,kernel_size=1,padding=0, ),)

        self.dim_ori_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, ),
            group_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5, kernel_size=1, padding=0, ), )

        self.keypoint_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, ),
            group_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, padding=0, ), )

        self.depth_head = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=3, padding=1, ),
            group_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0, ), )

        self.depth_hint_extractor1=nn.Conv2d(40, 1, kernel_size=1, padding=0)
        self.depth_hint_extractor2=nn.Conv2d(512, 1, kernel_size=1, padding=0)


    def forward(self, features,base_features):
        base_features = base_features.permute(0, 3, 1, 2)  # (batch_size,512,12,40) -> (batch_size,40,512,12)
        depth_hint = self.depth_hint_extractor1(base_features)  # (batch_size,40,512,12) -> (batch_size,1,512,12)
        depth_hint = depth_hint.permute(0,2,3,1)  # (batch_size,1,512,12) -> (batch_size,512,12,1)
        depth_hint=self.depth_hint_extractor2(depth_hint) # (batch_size,512,12,1) -> (batch_size,1,12,1)

        depth_hint_cat = depth_hint.repeat_interleave(8, dim=2) # (batch_size,1,12,1) -> (batch_size,1,96,1)
        depth_hint_cat = depth_hint_cat.expand(-1,-1,-1,320) # (batch_size,1,96,1) -> (batch_size,1,96,320)

        head_class = self.class_head(features)

        lstm_outputs,_=self.convGRU(features) # 2dcenter_offset, w&h, orientation, keypoint_offset, dimension_offset, depth_offset

        center2d_wh_reg=self.center2d_wh_head(lstm_outputs[0])
        dim_ori_reg=self.dim_ori_head(lstm_outputs[1])
        keypoint_reg=self.keypoint_head(lstm_outputs[2])
        depth_reg=self.depth_head(torch.cat([lstm_outputs[3],depth_hint_cat],dim=1))

        head_regression=torch.cat([depth_reg,keypoint_reg,dim_ori_reg,center2d_wh_reg],dim=1)

        head_class = sigmoid_hm(head_class)
        # (N, C, H, W)
        offset_dims = head_regression[:, self.dim_channel, ...].clone()
        head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

        vector_ori = head_regression[:, self.ori_channel, ...].clone()
        head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        return [head_class, head_regression]
