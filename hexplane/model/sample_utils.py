import torch
from torch.nn import functional as F
import hexplane.model.vf_utils as wru


def sample_from_hexplane(xyz_sampled, frame_time, plane, line_time, matMode, vecMode, align_corners, 
                         analytic_grad=False, multires_sample=False):
    # Prepare coordinates for grid sampling.
    # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
    if analytic_grad:
        grid_sample = wru.grid_sample # slow, pytorch-based (python) implementation supporting hessian
    else:
        grid_sample = F.grid_sample # fast, CUDNN-based implementation, but no hessian

    plane_coord = (
        torch.stack(
            (
                xyz_sampled[..., matMode[0]],
                xyz_sampled[..., matMode[1]],
                xyz_sampled[..., matMode[2]],
            )
        )
        .view(3, -1, 1, 2)
    )

    # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
    line_time_coord = torch.stack(
        (
            xyz_sampled[..., vecMode[0]],
            xyz_sampled[..., vecMode[1]],
            xyz_sampled[..., vecMode[2]],
        )
    )
    line_time_coord = (
        torch.stack(
            (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
        )
        .view(3, -1, 1, 2)
    )

    plane_feat, line_time_feat = [], []
    for idx_plane in range(len(plane)):
        # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
        plane_feat.append(
            grid_sample(
                plane[idx_plane],
                plane_coord[[idx_plane]],
                align_corners=align_corners,
            ).view(-1, *xyz_sampled.shape[:1])
        )
        # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
        line_time_feat.append(
            grid_sample(
                line_time[idx_plane],
                line_time_coord[[idx_plane]],
                align_corners=align_corners,
            ).view(-1, *xyz_sampled.shape[:1])
        )
    if multires_sample:
        plane_feats = [plane_feat]
        line_time_feats = [line_time_feat]

        plane_feat_ = []
        line_time_feat_ = []

        plane_feats.append(plane_feat_)
        line_time_feats.append(line_time_feat_)
        for idx_plane in range(len(plane)):
            plane_feat_.append(
                F.grid_sample(
                    plane[idx_plane][:, :, ::2, ::2],
                    plane_coord[[idx_plane]],
                    align_corners=align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_time_feat_.append(
                F.grid_sample(
                    line_time[idx_plane][:, :, ::2, ::2],
                    line_time_coord[[idx_plane]],
                    align_corners=align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat_ = []
        line_time_feat_ = []

        plane_feats.append(plane_feat_)
        line_time_feats.append(line_time_feat_)

        for idx_plane in range(len(plane)):
            plane_feat_.append(
                F.grid_sample(
                    plane[idx_plane][:, :, ::4, ::4],
                    plane_coord[[idx_plane]],
                    align_corners=align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_time_feat_.append(
                F.grid_sample(
                    line_time[idx_plane][:, :, ::4, ::4],
                    line_time_coord[[idx_plane]],
                    align_corners=align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat = [torch.concat([plane_feats[0][i], plane_feats[1][i], plane_feats[2][i]], dim=0) 
                      for i in range(len(plane_feats[0]))]
        line_time_feat = [torch.concat([line_time_feats[0][i], line_time_feats[1][i], line_time_feats[2][i]], dim=0) 
                      for i in range(len(line_time_feats[0]))]

    plane_feat, line_time_feat = (torch.cat(plane_feat), 
                                  torch.cat(line_time_feat))

    return plane_feat, line_time_feat