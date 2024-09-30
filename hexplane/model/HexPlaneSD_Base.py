from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur

from hexplane.model.HexPlane_Base import HexPlane_Base
from hexplane.model.HexPlane_Base import raw2alpha, DensityRender, RGBRender, SHRender
from hexplane.model.mlp import General_MLP, SceneFlow_MLP, SceneFlow_MLP_feat
from hexplane.model.TensoRF_Slim import TensoRF_Slim

import hexplane.model.vf_utils as wru


class HexPlaneSD_Base(HexPlane_Base):
    """
    A Static-Dynamic version of HexPlane, which supports different fusion methods and feature regressor methods.
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kwargs):

        self.depth_med_loss = kwargs.get("depth_med_loss", False) # use depth median loss

        self.stdensity_n_comp = kwargs.get("stdensity_n_comp", 8)
        self.stapp_n_comp = kwargs.get("stapp_n_comp", 24)
        
        self.stdensity_dim = kwargs.get("stdensity_dim", 1)
        self.stapp_dim = kwargs.get("stapp_dim", 27)

        self.dyst_entropy_skewness = kwargs.get("dyst_entropy_skewness", 1.0)

        # initialize base model (HexPlaneBase)
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kwargs)

        static_model = TensoRF_Slim(aabb, gridSize, device, time_grid, near_far, **kwargs)
        self.static_model = static_model

        self.update_static_model_params()

        self.ndc_system = kwargs.get("ndc_system", False)
    
    def update_static_model_params(self):
        # assign planes
        self.stapp_plane, self.stapp_line = (self.static_model.app_plane, 
                                             self.static_model.app_line_time)

        self.stdensity_plane, self.stdensity_line = (self.static_model.density_plane, 
                                                     self.static_model.density_line_time)
        
        # dynamic / static
        self.stdensity_basis_mat = self.static_model.density_basis_mat
        self.stapp_basis_mat = self.static_model.app_basis_mat

    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        self.density_plane, self.density_line_time = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        self.app_plane, self.app_line_time = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )

        if (
            self.fusion_two != "concat"
        ):  # if fusion_two is not concat, then we need dimensions from each paired planes are the same.
            assert self.app_n_comp[0] == self.app_n_comp[1]
            assert self.app_n_comp[0] == self.app_n_comp[2]

        # We use density_basis_mat and app_basis_mat to project extracted features from HexPlane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        if self.fusion_two == "concat":
            if self.fusion_one == "concat":
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp) * 2, self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp) * 2, self.app_dim, bias=False
                ).to(device)
            else:
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp), self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp), self.app_dim, bias=False
                ).to(device)
        else:
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        if self.DensityMode == "plain":
            # Initialize the basis matrices
            with torch.no_grad():
                weights = torch.ones_like(self.density_basis_mat.weight) / float(
                    self.density_dim
                )
                self.density_basis_mat.weight.copy_(weights)

    def init_one_hexplane(self, n_component, gridSize, device, time_grid=None):
        plane_coef, line_time_coef = [], []
        if time_grid is None:
            time_grid = self.time_grid
            
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], time_grid))
                    + self.init_shift
                )
            )

        return (torch.nn.ParameterList(plane_coef).to(device), 
                torch.nn.ParameterList(line_time_coef).to(device)
        )

    # will be used to set trainable variables.
    def get_optparam_groups(self, cfg, include_static=True, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]
        
        if include_static:
            # dynamic / static
            grad_vars += self.static_model.get_optparam_groups(cfg, lr_scale=lr_scale)
            
        return grad_vars

    def compute_densityfeature(
        self, xyz_sampled: torch.tensor, frame_time: torch.tensor, gaussian_grid_sample=False, analytic_grad=False
    ) -> torch.tensor:
        """
        Compuate the density features of sampled points from density HexPlane.

        Args:
            xyz_sampled: (B, N, 3) sampled points' xyz coordinates.
            frame_time: (B, N) sampled points' frame time.

        Returns:
            density: (B, N) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        if analytic_grad:
            grid_sample = wru.grid_sample # slow, pytorch-based (python) implementation supporting hessian
        else:
            grid_sample = F.grid_sample # fast, CUDNN-based implementation, but no hessian

        xyz_sampled = self.normalize_coord(xyz_sampled)

        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .view(3, -1, 1, 2)
            .detach()
        )

        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        ).detach()
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .view(3, -1, 1, 2)
        )

        if gaussian_grid_sample:
            raise NotImplementedError
        else:
            grid_sample = grid_sample
            grid_sample_t = grid_sample
        
        plane_feat, line_time_feat = [], []
        # Extract features from six feature planes.
        for idx_plane in range(len(self.density_plane)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                grid_sample(
                    self.density_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                grid_sample_t(
                    self.density_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_feat, line_time_feat = torch.cat(plane_feat, dim=0), torch.cat(
            line_time_feat, dim=0
        )

        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        if self.fusion_two == "multiply":
            raise NotImplementedError
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            raise NotImplementedError
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.density_basis_mat(inter.T)  # Feature Projection

        return inter
    
    def compute_appfeature(
        self, xyz_sampled: torch.tensor, frame_time, gaussian_grid_sample=False
    ) -> torch.tensor:
        """
        Compuate the app features of sampled points from appearance HexPlane.

        Args:
            xyz_sampled: (B, N, 3) sampled points' xyz coordinates.
            frame_time: (B, N) sampled points' frame time.

        Returns:
            density: (B, N) density of sampled points.
        """

        xyz_sampled = self.normalize_coord(xyz_sampled)

        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .view(3, -1, 1, 2)
        )

        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .view(3, -1, 1, 2)
        )

        if gaussian_grid_sample:
            grid_sample = self.grid_sample_gaussian
            grid_sample_t = self.grid_sample_gaussian_t
        else:
            grid_sample = F.grid_sample
            grid_sample_t = F.grid_sample

        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.app_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                grid_sample(
                    self.app_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                grid_sample_t(
                    self.app_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat, line_time_feat = torch.cat(plane_feat), torch.cat(
            line_time_feat
        )

        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        if self.fusion_two == "multiply":
            raise NotImplementedError
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            raise NotImplementedError
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.app_basis_mat(inter.T)  # Feature Projection

        return inter
    
    def compute_emptiness(self, xyz_locs, time_grid=64, length=1):
        """
        Compute the emptiness of spacetime points. Emptiness is the density.
        For each sapce point, we calcualte its densitis for various time steps and calculate its maximum density.
        """
        if self.emptyMask is not None:
            emptiness = self.emptyMask.sample_empty(xyz_locs)
            empty_mask = emptiness > 0
        else:
            empty_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if empty_mask.any():
            xyz_sampled = xyz_locs[empty_mask]
            xyz_sampled = self.denormalize_coord(xyz_sampled)
            time_samples = torch.linspace(-1, 1, time_grid).to(xyz_sampled.device)
            N, T = xyz_sampled.shape[0], time_samples.shape[0]
            xyz_sampled = (
                xyz_sampled.unsqueeze(1).expand(-1, T, -1).contiguous().view(-1, 3)
            )
            time_samples = (
                time_samples.unsqueeze(0).expand(N, -1).contiguous().view(-1, 1)
            )

            density_feature = self.compute_densityfeature(xyz_sampled, time_samples)
            stdensity_feature = self.static_model.compute_densityfeature(xyz_sampled, time_samples)
            sigma_feature = self.density_regressor(
                xyz_sampled,
                xyz_sampled,
                density_feature,
                time_samples,
            ).view(N, T)
            stsigma_feature = self.static_model.density_regressor(
                xyz_sampled,
                xyz_sampled,
                stdensity_feature,
                time_samples,
            ).view(N, T)

            sigma_feature = torch.amax(sigma_feature, -1)
            stsigma_feature = torch.amax(stsigma_feature, -1)
            validsigma = self.feature2density(sigma_feature)
            stvalidsigma = self.feature2density(stsigma_feature)
            sigma[empty_mask] = validsigma + stvalidsigma

        emptiness = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return emptiness
    

    def forward(
        self,
        rays_chunk: torch.tensor,
        frame_time: torch.tensor,
        white_bg: bool = True,
        is_train: bool = False,
        ndc_ray: bool = False,
        N_samples: int = -1,
        poses: dict = None
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, dict]:
        """
        Forward pass of the HexPlane.

        Args:
            rays_chunk: (B, 6) tensor, rays [rays_o, rays_d].
            frame_time: (B, 1) tensor, time values.
            white_bg: bool, whether to use white background.
            is_train: bool, whether in training mode.
            ndc_ray: bool, whether to use normalized device coordinates.
            N_samples: int, number of samples along each ray.
            poses: dict, pose matrix for each ray
        Returns:
            rgb: (B, 3) tensor, rgb values.
            depth: (B, 1) tensor, depth values.
            alpha: (B, 1) tensor, accumulated weights.
            z_vals: (B, N_samples) tensor, z values.
        """
        # Prepare rays.
        viewdirs = rays_chunk[:, 3:6]
        xyz_sampled, z_vals, ray_valid = self.sample_rays(
            rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
        )

        z_vals = torch.broadcast_to(z_vals, (xyz_sampled.shape[0], z_vals.shape[1]))
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1
        )
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
        
        assert ndc_ray == self.ndc_system
        if ndc_ray:
            dists = dists * rays_norm

        viewdirs = viewdirs / rays_norm

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        frame_time = frame_time.view(-1, 1, 1).expand(
            xyz_sampled.shape[0], xyz_sampled.shape[1], 1
        )

        # If emptiness mask is availabe, we first filter out rays with low opacities.
        if self.emptyMask is not None:
            xyz_sampled_normalized = self.normalize_coord(xyz_sampled[ray_valid])
            emptiness = self.emptyMask.sample_empty(xyz_sampled_normalized)
            empty_mask = emptiness > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~empty_mask
            ray_valid = ~ray_invalid

        # Initialize sigma and rgb values.
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        # initialize static sigma and static rgb values.
        stsigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        strgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        
        # Compute density feature and density if there are valid rays.
        if ray_valid.any():
            density_feature = self.compute_densityfeature(
                xyz_sampled[ray_valid], frame_time[ray_valid]
            )
            density_feature = self.density_regressor(
                xyz_sampled[ray_valid],
                viewdirs[ray_valid],
                density_feature,
                frame_time[ray_valid],
            )
            validsigma = self.feature2density(density_feature)
            sigma[ray_valid] = validsigma.view(-1)

            stdensity_feature = self.static_model.compute_densityfeature(
                xyz_sampled[ray_valid], frame_time[ray_valid]
            )
            stdensity_feature = self.static_model.density_regressor(
                xyz_sampled[ray_valid],
                viewdirs[ray_valid],
                stdensity_feature,
                frame_time[ray_valid],
            )
            stvalidsigma = self.static_model.feature2density(stdensity_feature)
            stsigma[ray_valid] = stvalidsigma.view(-1)
        
        # blend from static and dynamic densities
        alpha, stalpha, alpha_blended, weight, \
        stweight, weight_noblend, stweight_noblend, \
        T_blended = self.raw2outputs_blending(dists, sigma, stsigma)
        
        # Compute appearance feature and rgb if there are valid rays (whose weight are above a threshold).
        app_mask = weight_noblend > self.rayMarch_weight_thres
        stapp_mask = stweight_noblend > self.rayMarch_weight_thres

        # store auxiliary elements and returned
        aux = {}
        
        weight_blended = (weight + stweight)

        # probability that a point is dynamic rather than it is static
        dyblendw = sigma / (sigma + stsigma + 1e-16)

        # compute entropy
        aux['dyst_entropy'] = self.compute_entropy(dyblendw, skewness=self.dyst_entropy_skewness)
        aux['st_entropy'] = self.compute_static_entropy(stalpha, stsigma)

        # compute map
        dy_alpha_map = torch.sum(weight, dim=-1)
        st_alpha_map = torch.sum(stweight, dim=-1)

        aux['dy_alpha_map'] = dy_alpha_map
        aux['dy_alpha_mask'] = (dy_alpha_map > st_alpha_map).float()
        
        aux['st_alpha_map'] = st_alpha_map
        aux['st_alpha_mask'] = (st_alpha_map > dy_alpha_map).float()

        # compute static rgb values
        if app_mask.any():
            app_features = self.compute_appfeature(
                xyz_sampled[app_mask], frame_time[app_mask]
            )
            valid_rgbs = self.app_regressor(
                xyz_sampled[app_mask],
                viewdirs[app_mask],
                app_features,
                frame_time[app_mask],
            )
            rgb[app_mask] = valid_rgbs

        if stapp_mask.any():
            stapp_features = self.static_model.compute_appfeature(
                xyz_sampled[stapp_mask], frame_time[stapp_mask]
            )
            stvalid_rgbs = self.static_model.app_regressor(
                xyz_sampled[stapp_mask],
                viewdirs[stapp_mask],
                stapp_features,
                frame_time[stapp_mask],
            )
            strgb[stapp_mask] = stvalid_rgbs

        if is_train:
            xyz_post = xyz_sampled[app_mask]
            xyz_prev = xyz_sampled[app_mask]
            
            # render optical flow (forward) at the next pose
            if poses is not None and "post_pose" in poses.keys():
                aux['optical_flow_fw_post'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_post, poses['pose'], poses['post_pose'], weight, stweight)

            # render optical flow (backward) at the prev pose 
            if poses is not None and "prev_pose" in poses.keys():
                aux['optical_flow_bw_prev'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_prev, poses['pose'], poses['prev_pose'], weight, stweight)

            # render optical flow (forward and backward) at the ref pose
            if poses is not None and "pose" in poses.keys():
                aux['optical_flow_fw_ref'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_post, poses['pose'], poses['pose'], weight, stweight)
                aux['optical_flow_bw_ref'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_prev, poses['pose'], poses['pose'], weight, stweight)
        
        acc_map = torch.sum(weight_noblend, -1)
        rgb_map = torch.sum(weight_noblend[..., None] * rgb, -2)

        stacc_map = torch.sum(stweight_noblend, -1)
        strgb_map = torch.sum(stweight_noblend[..., None] * strgb, -2)

        acc_map_blended = torch.sum(weight_blended, -1)

        rgb_blended = (rgb * weight[...,None] + strgb * stweight[...,None])
        rgb_map_blended = torch.sum(rgb_blended, -2)
        
        # If white_bg or (is_train and torch.rand((1,))<0.5):
        if white_bg or not is_train:
            rgb_map_blended = rgb_map_blended + (1.0 - acc_map_blended[..., None])
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
            strgb_map = strgb_map + (1.0 - stacc_map[..., None])

            rgb_map_blended = rgb_map_blended.clamp(0, 1)
            rgb_map = rgb_map.clamp(0, 1)
            strgb_map = strgb_map.clamp(0, 1)
        else:
            rgb_map_blended = rgb_map_blended + (1.0 - acc_map_blended[..., None]) * torch.rand(
                size=(1, 3), device=rgb_map_blended.device
            )
            rgb_map = rgb_map + (1.0 - acc_map[..., None]) * torch.rand(
                size=(1, 3), device=rgb_map.device
            )
            strgb_map = strgb_map + (1.0 - stacc_map[..., None]) * torch.rand(
                size=(1, 3), device=strgb_map.device
            )

            rgb_map_blended = rgb_map_blended.clamp(0, 1)
            rgb_map = rgb_map.clamp(0, 1)
            strgb_map = strgb_map.clamp(0, 1)

        aux['dyrgb_map'] = rgb_map
        aux['strgb_map'] = strgb_map

        # Calculate depth.
        if self.depth_loss or self.depth_med_loss:
            dydepth_map = torch.sum(weight_noblend * z_vals, -1)
            dydepth_map = dydepth_map + (1.0 - acc_map) * rays_chunk[..., -1]

            stdepth_map = torch.sum(stweight_noblend * z_vals, -1)
            stdepth_map = stdepth_map + (1.0 - stacc_map) * rays_chunk[..., -1]

            aux['dydepth_map'] = dydepth_map
            aux['stdepth_map'] = stdepth_map

            depth_map_blended = torch.sum(weight_blended * z_vals, -1)
            depth_map_blended = depth_map_blended + (1.0 - acc_map_blended) * rays_chunk[..., -1]
        else:
            with torch.no_grad():
                depth_map = torch.sum(weight_noblend * z_vals, -1)
                depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]

                stdepth_map = torch.sum(stweight_noblend * z_vals, -1)
                stdepth_map = stdepth_map + (1.0 - stacc_map) * rays_chunk[..., -1]

                aux['dydepth_map'] = depth_map
                aux['stdepth_map'] = stdepth_map

                depth_map_blended = torch.sum(weight_blended * z_vals, -1)
                depth_map_blended = depth_map_blended + (1.0 - acc_map_blended) * rays_chunk[..., -1]
        return rgb_map_blended, depth_map_blended, weight_blended, z_vals, aux

    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx]) + reg2(self.density_line_time[idx])
            )
        return total
    
    def TV_loss_stdensity(self, reg, reg2=None):
        total = self.static_model.TV_loss_density(reg, reg2)
        return total

    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) + reg2(self.app_line_time[idx])
        return total
    
    def TV_loss_stapp(self, reg, reg2=None):
        total = self.static_model.TV_loss_app(reg, reg2)
        return total

    def L1_loss_density(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        return total

    def L1_loss_app(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        return total
    
    def L1_loss_stdensity(self):
        return self.static_model.L1_loss_density()

    def L1_loss_stapp(self):
        return self.static_model.L1_loss_app()

    @torch.no_grad()
    def getDenseEmpty(self, gridSize=None, time_grid=None):
        """
        For a 4D volume, we sample the opacity values of discrete spacetime points and store them in a 3D volume.
        Note that we always assume the 4D volume is in the range of [-1, 1] for each axis.
        """
        gridSize = self.gridSize if gridSize is None else gridSize
        time_grid = self.time_grid if time_grid is None else time_grid

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = samples * 2.0 - 1.0
        emptiness = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            emptiness[i] = self.compute_emptiness(
                dense_xyz[i].view(-1, 3).contiguous(), time_grid, self.stepSize
            ).view((gridSize[1], gridSize[2]))
        return emptiness, dense_xyz

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, line_time_coef, res_target, time_grid):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
            line_time_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_time_coef[i].data,
                    size=(res_target[vec_id], time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef, line_time_coef
    
    @torch.no_grad()
    def up_sampling_planeline(self, plane_coef, line_coef, res_target, time_grid):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_coef[i].data,
                    size=(res_target[vec_id], 1),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")

    def denormalize_coord(self, xyz_sampled):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """

        if self.normalize_type == "normal":
            aabb = torch.FloatTensor(self.aabb).to(self.device)
            invaabbSize = self.invaabbSize.clone().to(self.device)
            return (xyz_sampled + 1.) / invaabbSize + aabb[0]
        
    def raw2outputs_blending(self, dists, dysigma, stsigma):
        dists = dists * self.distance_scale

        op = torch.exp(-dysigma * dists)
        stop = torch.exp(-stsigma * dists)
        
        dyalpha = 1.0 - op
        stalpha = 1.0 - stop
        alpha_blended = 1.0 - torch.exp(-(dysigma + stsigma) * dists)

        T_dy = torch.cumprod(
            torch.cat(
                [torch.ones(dyalpha.shape[0], 1).to(dyalpha.device), (1.0 - dyalpha) + 1e-10], -1
            ),
            -1,
        )
        
        T_st = torch.cumprod(
            torch.cat(
                [torch.ones(stalpha.shape[0], 1).to(stalpha.device), (1.0 - stalpha) + 1e-10], -1
            ),
            -1,
        )

        T_blended = torch.cumprod(
            torch.cat(
                [torch.ones(alpha_blended.shape[0], 1).to(alpha_blended.device), (1.0 - alpha_blended) + 1e-10], -1
            ),
            -1,
        )

        dyweights = dyalpha * T_blended[:, :-1]  # [N_rays, N_samples]
        stweights = stalpha * T_blended[:, :-1]  # [N_rays, N_samples]

        dyweights_noblned = dyalpha * T_dy[:, :-1]
        stweights_noblned = stalpha * T_st[:, :-1]

        return dyalpha, stalpha, alpha_blended, \
               dyweights, stweights, dyweights_noblned, stweights_noblned, T_blended[:, :-1]

    def compute_entropy(self, blendw, clip_threshold=1e-16, skewness=1.0):
        blendw = torch.clamp(blendw ** skewness, clip_threshold, 1. - clip_threshold)
        rev_blendw = torch.clamp(1.-blendw, clip_threshold, 1.)
        entropy = - (blendw * torch.log(blendw) + rev_blendw * torch.log(rev_blendw))
        return entropy

    def compute_static_entropy(self, stalpha, stsigma, mask_thresold=0.1, clip_threshold=1e-16):
        stalpha = torch.clamp(stalpha, clip_threshold)
        p = stalpha / torch.sum(stalpha, dim=-1, keepdim=True) 

        stsigma_sum = torch.sum(stsigma, dim=-1, keepdim=True) 
        mask = (stsigma_sum >= mask_thresold) 
        
        entropy = mask * -torch.mean(p * torch.log(p), dim=-1, keepdim=True)
        
        return entropy

    def grid_sample_gaussian(self, input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        input_gaussian = self.gaussian_blur(input)
        sampled = F.grid_sample(
            input=input_gaussian,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners
        )

        return sampled
    
    def grid_sample_gaussian_t(self, input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        input_gaussian = self.gaussian_blur_t(input)
        sampled = F.grid_sample(
            input=input_gaussian,
            grid=grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners
        )

        return sampled
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the original state_dict
        original_state_dict = super().state_dict(destination, prefix, keep_vars)
        
        new_state_dict = {k: v for k, v in original_state_dict.items() if 'static_model' not in k}
        
        return new_state_dict
    
    def render_optical_flow(self, xyz, app_mask, xyz_targ, pose_ref, pose_targ, dyweight, stweight):
        xyz_all = xyz.clone().detach()
        xyz_all[app_mask] = xyz_targ
        ndc_params = self.ndc_params
        
        if self.ndc_system:
            optflow = wru.compute_optical_flow(pose_ref, pose_targ, 
                                           ndc_params['H'], ndc_params['W'], ndc_params['focal'],
                                           dyweight, stweight, xyz, 
                                           xyz_all, center=ndc_params['center'])
        else:
            optflow = wru.compute_optical_flow_world(pose_ref, pose_targ,
                                            ndc_params['H'], ndc_params['W'], ndc_params['focal'],
                                            dyweight, stweight, xyz, 
                                            xyz_all, center=ndc_params['center'])
        optflow[torch.isnan(optflow)] = 0.
        optflow = torch.clamp(optflow, -400., 400.)
        return optflow