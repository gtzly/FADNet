import numpy as np

import torch

class OutputDecoder():
    def __init__(self, dim_templates):
        self.dim_templates = torch.as_tensor(dim_templates).to(device="cuda")

    @staticmethod
    def rad_to_matrix(rotys, N):
        device = rotys.device

        cos, sin = rotys.cos(), rotys.sin()

        i_temp = torch.tensor([[1, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 1]]).to(dtype=torch.float32,
                                               device=device)
        ry = i_temp.repeat(N, 1).view(N, -1, 3)

        ry[:, 0, 0] *= cos
        ry[:, 0, 2] *= sin
        ry[:, 2, 0] *= sin
        ry[:, 2, 2] *= cos

        return ry

    def decode_depth(self, depths_offset):
        depth = 1. / (depths_offset.sigmoid() + 1e-6) - 1.

        return depth

    def decode_location(self,points,points_offset,depths,calibs,trans_mats):
        device = points.device

        calibs = calibs.to(device=device)
        trans_mats = trans_mats.to(device=device)

        # number of points
        N = points_offset.shape[0]
        # batch size
        N_batch = calibs.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()

        trans_mats_inv = trans_mats.inverse()[obj_id]
        calibs_inv = calibs.inverse()[obj_id]

        points = points.view(-1, 2)
        assert points.shape[0] == N
        proj_points = points.float() + points_offset
        # transform project points in homogeneous form.
        proj_points_extend = torch.cat(
            (proj_points, torch.ones(N, 1).to(device=device)), dim=1)
        # expand project points as [N, 3, 1]
        proj_points_extend = proj_points_extend.unsqueeze(-1)
        # transform project points back on image
        proj_points_img = torch.matmul(trans_mats_inv, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * depths.view(N, -1, 1)
        # transform image coordinates back to object locations
        locations = torch.matmul(calibs_inv, proj_points_img)

        return locations.squeeze(2)

    def decode_dimension(self, cls_id, dims_offset):
        cls_id = cls_id.flatten().long()

        dims_select = self.dim_templates[cls_id, :]
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def decode_orientation(self, vector_ori, locations):
        locations = locations.view(-1, 3)
        rays = torch.atan2(locations[:, 0] , (locations[:, 2] + 1e-7))
        alphas = torch.atan2(vector_ori[:, 0] , (vector_ori[:, 1] + 1e-7))

        rotys = alphas + rays

        larger_idx = (rotys > np.pi).nonzero()
        small_idx = (rotys < -np.pi).nonzero()

        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        return rotys, alphas

    def decode_2d(self,points_3d,points_2d_offsets,dims_2d,trans_mats):
        device = points_3d.device
        trans_mats = trans_mats.to(device=device)

        points_3d = points_3d.view(-1, 2)
        points_2d_offsets = points_2d_offsets.view(-1, 2)

        # number of points
        N = points_2d_offsets.shape[0]
        # batch size
        N_batch = trans_mats.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()

        trans_mats_inv = trans_mats.inverse()[obj_id]

        assert points_3d.shape[0] == N 
        proj_2d_points = points_3d.float() + points_2d_offsets 
        dims_2d=dims_2d.view(-1,2) 
        top_left = proj_2d_points - dims_2d / 2.            
        bottom_right = proj_2d_points + dims_2d / 2.         

        # transform project points in homogeneous form.
        top_left_extend = torch.cat(
            (top_left, torch.ones(N, 1).to(device=device)), dim=1)
        bottom_right_extend = torch.cat(
            (bottom_right, torch.ones(N, 1).to(device=device)), dim=1)

        top_left_extend = top_left_extend.unsqueeze(-1)
        bottom_right_extend = bottom_right_extend.unsqueeze(-1)

        # transform project points back on image
        top_left_img = torch.matmul(trans_mats_inv, top_left_extend).squeeze(-1)[:, :2]
        bottom_right_img = torch.matmul(trans_mats_inv, bottom_right_extend).squeeze(-1)[:, :2]

        bbox_corner=torch.cat((top_left_img,bottom_right_img),dim=1) # （N，4）

        return bbox_corner
