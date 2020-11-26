import torch
from torch import nn

from lib.utils.heatmap_op import (
    nms_hm,
    select_topk,
    select_point_of_interest,
)


class PostProcessor(nn.Module):
    def __init__(self,output_coder,det_threshold,max_detection):
        super(PostProcessor, self).__init__()
        self.output_coder = output_coder
        self.det_threshold = det_threshold
        self.max_detection = max_detection

    def prepare_meta(self, targets):
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        calib = torch.stack([t.get_field("calib") for t in targets])
        size = torch.stack([torch.tensor(t.size) for t in targets])

        return dict(trans_mat=trans_mat,
                    calib=calib,
                    size=size)

    def forward(self, predictions, meta_data):
        pred_heatmap, pred_regression = predictions
        batch = pred_heatmap.shape[0]

        meta = self.prepare_meta(meta_data)

        heatmap = nms_hm(pred_heatmap)

        scores, indexs, clses, ys, xs = select_topk(
            heatmap,
            K=self.max_detection,
        )

        pred_regression = select_point_of_interest(
            batch, indexs, pred_regression
        )
        pred_regression_pois = pred_regression.view(-1, 12)
        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)

        keep_idx = (scores.squeeze() > self.det_threshold)

        if sum(keep_idx)==0:
            result = torch.Tensor()
            return result

        pred_regression_pois=pred_regression_pois[keep_idx]
        pred_proj_points=pred_proj_points[keep_idx]
        clses = clses.squeeze()[keep_idx] #  -> (2,)
        scores = scores.squeeze()[keep_idx] #  -> (2,)

        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:8]
        pred_center_2d_offsets = pred_regression_pois[:, 8:10]
        pred_dims_2d = pred_regression_pois[:, 10:12]

        pred_depths = self.output_coder.decode_depth(pred_depths_offset) # (2,)
        pred_locations = self.output_coder.decode_location(  #(2,3)
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            meta["calib"],
            meta["trans_mat"]
        )
        pred_dimensions = self.output_coder.decode_dimension(
            clses,
            pred_dimensions_offsets
        )
        # translate 3D center to bottom center
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.output_coder.decode_orientation(
            pred_orientation,
            pred_locations
        )
        bbox = self.output_coder.decode_2d(pred_proj_points,
                                             pred_center_2d_offsets,
                                             pred_dims_2d,
                                             meta['trans_mat'])

        # lhw -> hwl
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

        result = torch.cat([
            clses.unsqueeze(-1), pred_alphas.unsqueeze(-1), bbox,
            pred_dimensions,pred_locations, pred_rotys.unsqueeze(-1),
            scores.unsqueeze(-1)], dim=1)

        return result