from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

import numpy as np

from hexplane.model.HexPlaneSD_Base import HexPlaneSD_Base
from hexplane.model.mlp import SceneFlow_MLP_feat

import hexplane.model.vf_utils as vfu
from hexplane.model.gradient_scale import GradientScale
from hexplane.model.sample_utils import sample_from_hexplane


class HexPlaneSD_Flow(HexPlaneSD_Base):
    """
    A SD version of HexPlane, which supports different fusion methods and feature regressor methods.
    SD stands for Static/Dynamic. And this is the 'Flow' added version of the HexPlaneSD.
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kwargs):
        # Flow related params added to HexPlane_Base
        self.flow_n_comp = kwargs.get("flow_n_comp", 24) 
        self.flow_dim = kwargs.get("flow_dim", 27)
        self.flow_t_pe = kwargs.get("flow_t_pe", -1)
        self.flow_pos_pe = kwargs.get("flow_pos_pe", -1)
        self.flow_view_pe = kwargs.get("flow_view_pe", -1)
        self.flow_fea_pe = kwargs.get("flow_fea_pe", -1)
        self.flow_featureC = kwargs.get("flow_featureC", 128)
        self.flow_n_layers = kwargs.get("flow_n_layers", 3)

        # whether to compute hessian
        self.compute_hessian = True
        
        # flow mode and the number of hessian samples; the latter is used only for computing strain rates. 
        self.FlowMode = kwargs.get("FlowMode", "general_MLP")
        self.num_hessian_samples = kwargs.get("num_hessian_samples", 32)
        self.hessian_sample_type = kwargs.get("hessian_sample_type", "ray")

        # compute depth map to use depth med loss
        self.depth_med_loss = kwargs.get("depth_med_loss", False)

        # for ndc <-> world conversion and query different time frames
        self.ndc_params = kwargs['ndc_params']
        self.time_interval = kwargs['time_interval']
        self.max_timeinter = kwargs.get("init_timeinter", 1.0) # set max_timeinter to init_timeinter
        self.max_timehop = kwargs.get("init_timehop", 1.0) # set max_timeinter to init_timeinter

        # debug velocity field (additional outputs)
        self.render_kinematic_field = False
        self.return_xyz = False

        # flow order (if it is 0, it is NSFF style discrete flow formulation: backward and forward)
        self.flow_order = kwargs.get("flow_order", 3)

        # multi-resolution sampling strategy (similar to RoDyNRF)
        self.multires_sample = kwargs.get("multires_sample", False)

        # numerical gradient resolution scale (scale > 1 means larger flow resolution)
        self.num_grad_res_scale = kwargs.get("num_grad_res_scale", 1.0)

        # initialize parent model
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kwargs)

        # taylor mask
        self.use_taylor_order_mask = kwargs.get("use_taylor_order_mask", False)
        if self.flow_order == 0:
            self.use_taylor_order_mask = torch.ones(2, dtype=torch.float32)
        elif self.use_taylor_order_mask:
            taylor_mask = torch.zeros(self.flow_order, dtype=torch.float32, device=self.device)
            self.register_buffer('taylor_mask', taylor_mask)
            self.taylor_mask[0] = 1.
        else:
            self.taylor_mask = torch.ones(self.flow_order, dtype=torch.float32)

        # intialize flow function
        self.init_flow_func(self.FlowMode,
                            self.flow_dim,
                            self.flow_order,
                            self.flow_t_pe,
                            self.flow_pos_pe,
                            self.flow_view_pe,
                            self.flow_fea_pe,
                            self.flow_featureC,
                            self.flow_n_layers,
                            self.device,
                          )

        # flow gradient scaler
        self.flowgrad_scale = kwargs.get("flowgrad_scale", 1.0)
        self.flowgrad_scaler = GradientScale(self.flowgrad_scale)

        # density gradient scaler
        self.densitygrad_scaler = GradientScale(0.1)

        # velocity gradient scaler
        self.velocitygrad_scaler = GradientScale(0.1)

        # density gradient scaler
        self.dyalphagrad_scaler = GradientScale(0.01)

        # flow scaler
        self.flow_scale = kwargs.get("flow_scale", 1.0)

        # epsilon multiplier (larger means smaller epsilon)
        self.grad_res_multiplier = kwargs.get("grad_res_multiplier", 5.0)

        if self.ndc_system:
            sigma_scale = vfu.ndc2world_sigma(torch.ones(1, device=self.device, dtype=torch.float32), 
                                            torch.zeros(3, device=self.device, dtype=torch.float32), 
                                            self.ndc_params['H'], self.ndc_params['W'],
                                            self.ndc_params['focal'], self.ndc_params['near'], 0.0)
            self.sigma_scale = sigma_scale.detach().cpu().item()
        else:
            self.sigma_scale = 1.0

        self.use_viewdir_dy = True

        self.compute_flow = True
    
    # initialize flow regressor
    def init_flow_func(
        self, FlowMode, flow_dim, flow_order, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device
    ):
        """
        Initialize density regression function.
        """
        if flow_order > 0:
            out_channel = 3 * flow_order
        else:
            out_channel = 6 # backward 3 + forward 3

        if FlowMode == "general_MLP":  # Use general MLP to estimate flow.
            self.flow_regressor = SceneFlow_MLP_feat(
                flow_dim,
                out_channel,
                t_pe,
                fea_pe,
                pos_pe,
                view_pe,
                featureC,
                n_layers,
                zero_init=True # self.use_ndc_to_world,
            ).to(device)
        else:
            raise NotImplementedError("No such Flow Regression Mode")
        
        # General MLP to estimate normal.
        self.normal_regressor = SceneFlow_MLP_feat(
            flow_dim,
            3,
            t_pe,
            fea_pe,
            pos_pe,
            -1,
            32,
            2,
            zero_init=False
        ).to(device)

        print("FLOW REGRESSOR:")
        print(self.flow_regressor)

    # will be used to set trainable variables. Also support getting flow-related variables separately.
    def get_optparam_groups(self, cfg, lr_scale=1.0, include_static=True, include_flow=True, only_flow=False):
        grad_vars = super().get_optparam_groups(cfg, include_static=include_static, lr_scale=lr_scale)

        if include_flow:
            if only_flow:
                grad_vars = []
            grad_vars += [
                {
                    "params": self.flow_line_time,
                    "lr": lr_scale * cfg.lr_flow_grid,
                    "lr_org": cfg.lr_flow_grid,
                },
                {
                    "params": self.flow_plane,
                    "lr": lr_scale * cfg.lr_flow_grid,
                    "lr_org": cfg.lr_flow_grid,
                },
                {
                    "params": self.flow_basis_mat.parameters(),
                    "lr": lr_scale * cfg.lr_flow_nn,
                    "lr_org": cfg.lr_flow_nn,
                },
            ]

            if isinstance(self.flow_regressor, torch.nn.Module):
                grad_vars += [
                    {
                        "params": self.flow_regressor.parameters(),
                        "lr": lr_scale * cfg.lr_flow_nn,
                        "lr_org": cfg.lr_flow_nn,
                    }
                ]

        return grad_vars

    # initialize each plane. Note that flow-related "flow_plane" is added.
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
            dim_multiplier = 1
            if self.fusion_one == "concat":
                dim_multiplier *= 2
            if self.multires_sample:
                dim_multiplier *= 3
            self.density_basis_mat = torch.nn.Linear(
                sum(self.density_n_comp) * dim_multiplier, self.density_dim, bias=False
            ).to(device)
            self.app_basis_mat = torch.nn.Linear(
                sum(self.app_n_comp) * dim_multiplier, self.app_dim, bias=False
            ).to(device)
        else:
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        # Initialize the basis matrices
        with torch.no_grad():
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            self.density_basis_mat.weight.copy_(weights)
        
        # flow plane
        flow_grid_size = [int(s * self.num_grad_res_scale) for s in self.gridSize]
        flow_time_grid = int(self.time_grid * self.num_grad_res_scale)
        self.flow_plane, self.flow_line_time = self.init_one_hexplane(
            self.flow_n_comp, flow_grid_size, device, time_grid=flow_time_grid
        )

        # We use density_basis_mat and app_basis_mat to project extracted features from HexPlane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        if self.fusion_two == "concat":
            if self.fusion_one == "concat":
                self.flow_basis_mat = torch.nn.Linear(
                    sum(self.flow_n_comp) * 2, self.flow_dim, bias=False
                ).to(device)
            else:
                self.flow_basis_mat = torch.nn.Linear(
                    sum(self.flow_n_comp), self.flow_dim, bias=False
                ).to(device)
        else:
            self.flow_basis_mat = torch.nn.Linear(
                self.flow_n_comp[0], self.flow_dim, bias=False
            ).to(device)

        print("plane sizes:")
        print("  app_plane: ")
        for p, v in zip(self.app_plane, self.app_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  density_plane: ")
        for p, v in zip(self.density_plane, self.density_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  flow_plane: ")
        for p, v in zip(self.flow_plane, self.flow_line_time):
            print("  - %s, %s" % (p.shape, v.shape))

    # sample flow features from the flow planes
    def compute_flowfeature(
        self, xyz_sampled: torch.tensor, frame_time: torch.tensor
    ) -> torch.tensor:
        """
        Compuate the flow features of sampled points from flow HexPlane.

        Args:
            xyz_sampled: (B, N, 3) sampled points' xyz coordinates.
            frame_time: (B, N) sampled points' frame time.
        Returns:
            inter: (B, N, dim) feature for a flow component 
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        grid_sample = F.grid_sample # fast, CUDNN-based implementation, but no hessian

        xyz_sampled = self.normalize_coord(xyz_sampled)
        
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

        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.flow_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                grid_sample(
                    self.flow_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                grid_sample(
                    self.flow_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        if self.fusion_two == "concat":
            plane_feat, line_time_feat = torch.cat(plane_feat, dim=0), torch.cat(
                line_time_feat, dim=0
            )
        else:
            plane_feat, line_time_feat = torch.stack(plane_feat, dim=0), torch.stack(
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
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.flow_basis_mat(inter.T)  # Feature Projection

        return inter
    
    def compute_densityfeature(
        self, xyz_sampled: torch.tensor, frame_time: torch.tensor
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

        xyz_sampled = self.normalize_coord(xyz_sampled)

        plane_feat, line_time_feat = sample_from_hexplane(xyz_sampled, 
                                        frame_time, self.density_plane, 
                                        self.density_line_time, self.matMode, self.vecMode, self.align_corners, 
                                        multires_sample=self.multires_sample)

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
        self, xyz_sampled: torch.tensor, frame_time
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

        plane_feat, line_time_feat = sample_from_hexplane(xyz_sampled, 
                                        frame_time, self.app_plane, 
                                        self.app_line_time, self.matMode, self.vecMode, self.align_corners, 
                                        multires_sample=self.multires_sample)

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

    def compute_sf_bundle_grad(self, xyz_sampled, viewdirs, frame_time, is_train=False, **kwargs):
        if self.ndc_system:
            ndc_params = [self.ndc_params['H'], 
                            self.ndc_params['W'], 
                            self.ndc_params['focal'],
                            self.ndc_params['near'], 0.0]

        res = self.compute_sf_bundle(xyz_sampled, viewdirs, frame_time, get_pred_normal=True, **kwargs)
        if self.ndc_system:
            res['sigma_world'] = vfu.ndc2world_sigma(res['sigma'], xyz_sampled.detach(), *ndc_params)

        resolutions = [self.flow_line_time[2].shape[2],  # resolution of x-axis
                        self.flow_line_time[1].shape[2],  # resolution of y-axis
                        self.flow_line_time[0].shape[2]]  # resolution of z-axis
        resolution_t = self.flow_line_time[0].shape[3]   # resolution of t-axis
        
        multiplier = self.nSamples / resolutions[2]
        multiplier *= self.grad_res_multiplier
        if self.align_corners:
            epsilons = 2. / (np.array(resolutions) * multiplier * 1.0)  # from normalized one (-1 ~ 1)
            epsilon_t = 2. / (resolution_t * multiplier)
        else:
            epsilons = 2. / (np.array(resolutions) * multiplier * 1.0 - 1.) 
            epsilon_t = 2. / (resolution_t * multiplier - 1.)

        if is_train:
            interep = np.random.rand(*epsilons.shape)
            interep_t = np.random.rand()

            epsilons = epsilons * (interep + 0.5)
            epsilon_t = epsilon_t * (interep_t + 0.5)
        
        epsilons = torch.FloatTensor(epsilons).to(xyz_sampled.device)

        # normalized coordinates
        xyz_sampled_n = self.normalize_coord(xyz_sampled)

        # clamping is required to prevent points over boundaries
        xyz_sampled_nl = torch.clamp(xyz_sampled_n - epsilons, -1, 1)
        xyz_sampled_nu = torch.clamp(xyz_sampled_n + epsilons, -1, 1)
        

        if self.ndc_system:
            xyz_world = vfu.ndc2world(xyz_sampled, *ndc_params)

            epsilon_list = []
            for i in range(3):
                xyz_sampled_u = xyz_sampled_n.clone()
                xyz_sampled_l = xyz_sampled_n.clone()
                xyz_sampled_u[...,0] = xyz_sampled_nu[...,i].clone() # change only one axis (i-axis)
                xyz_sampled_u = self.denormalize_coord(xyz_sampled_u)
                xyz_sampled_l[...,0] = xyz_sampled_nl[...,i].clone() # change only one axis (i-axis)
                xyz_sampled_l = self.denormalize_coord(xyz_sampled_l)
                xyz_world_u = vfu.ndc2world(xyz_sampled_u, *ndc_params)
                xyz_world_l = vfu.ndc2world(xyz_sampled_l, *ndc_params)
                epsilons = torch.linalg.norm(xyz_world_u - xyz_world_l, dim=-1) / 2.0
                epsilons_sorted, _ = torch.sort(epsilons)
                epsilons = torch.clamp(epsilons, epsilons_sorted[0], epsilons_sorted[int(epsilons.shape[0]*0.95)])
                epsilon_list.append(epsilons)
            epsilons = torch.min(torch.stack(epsilon_list, dim=0), dim=0, keepdim=False)[0]
            

            xyz_world_us = []
            xyz_world_ls = []
            for i in range(3):
                xyz_world_u = xyz_world.clone()
                xyz_world_u[...,i] = xyz_world[..., i] + epsilons
                xyz_world_l = xyz_world.clone()
                xyz_world_l[...,i] = xyz_world[..., i] - epsilons

                xyz_world_us.append(xyz_world_u)
                xyz_world_ls.append(xyz_world_l)

        frame_time_nl = torch.clamp(frame_time - epsilon_t, -1, 1) # note that frame_time is already (-1, 1)
        frame_time_nu = torch.clamp(frame_time + epsilon_t, -1, 1)
            
        # compute gradients w.r.t x, y, z
        spatial_grads = []
        spatial_grads_sigma = []
        valid_mask = torch.ones_like(xyz_sampled[:,0])
        for i in range(3):
            if self.ndc_system:
                xyz_sampled_u = vfu.world2ndc(xyz_world_us[i], *ndc_params)
                xyz_sampled_l = vfu.world2ndc(xyz_world_ls[i], *ndc_params)

                aabb = torch.FloatTensor(self.aabb).to(xyz_sampled_u.device)

                valid_mask_u = torch.logical_and(xyz_sampled_u > aabb[0], xyz_sampled_u < aabb[1])
                valid_mask_l = torch.logical_and(xyz_sampled_l > aabb[0], xyz_sampled_l < aabb[1])
                valid_mask_ = torch.logical_and(valid_mask_u, valid_mask_l)
                valid_mask_ = torch.all(valid_mask, dim=-1)
                valid_mask = torch.logical_and(valid_mask, valid_mask_)
                denom = xyz_world_us[i][...,i:i+1] - xyz_world_ls[i][...,i:i+1]
            else:
                xyz_sampled_u = xyz_sampled_n.clone() # reset u
                xyz_sampled_l = xyz_sampled_n.clone() # reset l

                xyz_sampled_u[...,i] = xyz_sampled_nu[...,i].clone() # change only one axis (i-axis)
                xyz_sampled_u = self.denormalize_coord(xyz_sampled_u)
                xyz_sampled_l[...,i] = xyz_sampled_nl[...,i].clone() # change only one axis (i-axis)
                xyz_sampled_l = self.denormalize_coord(xyz_sampled_l)
                denom = xyz_sampled_u[...,i:i+1] - xyz_sampled_l[...,i:i+1]
                
            res_u = self.compute_sf_bundle(xyz_sampled_u.detach(), viewdirs, frame_time, **kwargs)
            res_l = self.compute_sf_bundle(xyz_sampled_l.detach(), viewdirs, frame_time, **kwargs)

            diff = res_u['flows_ref'] - res_l['flows_ref']
            
            sigma_u = res_u['sigma'] / 16.0
            sigma_l = res_l['sigma'] / 16.0

            if self.ndc_system:
                sigma_u = vfu.ndc2world_sigma(sigma_u, xyz_sampled_u.detach(), *ndc_params)
                sigma_l = vfu.ndc2world_sigma(sigma_l, xyz_sampled_l.detach(), *ndc_params)

            diff_sigma = sigma_u - sigma_l
            grad = diff / (denom.detach() + 1e-10)
            grad_sigma = diff_sigma / (denom[...,0].detach() + 1e-10)
            spatial_grads.append(grad)
            spatial_grads_sigma.append(grad_sigma)

        frame_time_u = frame_time_nu
        frame_time_l = frame_time_nl

        res_u = self.compute_sf_bundle(xyz_sampled.detach(), viewdirs, frame_time_u, **kwargs)
        res_l = self.compute_sf_bundle(xyz_sampled.detach(), viewdirs, frame_time_l, **kwargs)

        diff = res_u['flows_ref'] - res_l['flows_ref']
        sigma_u = res_u['sigma'] / 16.0
        sigma_l = res_l['sigma'] / 16.0

        if self.ndc_system:
            sigma_u = vfu.ndc2world_sigma(sigma_u, xyz_sampled.detach(), *ndc_params)
            sigma_l = vfu.ndc2world_sigma(sigma_l, xyz_sampled.detach(), *ndc_params)

        diff_sigma = sigma_u - sigma_l
        denom = (frame_time_nu - frame_time_nl)

        grad_t = diff / (denom.detach() + 1e-10)
        grad_t_sigma = diff_sigma / (denom[...,0].detach() + 1e-10)

        if self.ndc_system:
            sNdc_dX = vfu.jacobian_world2ndc(xyz_sampled, self.ndc_params['H'], 
                                                self.ndc_params['W'], 
                                                self.ndc_params['focal'],
                                                self.ndc_params['near'], epsilon=0.0).detach()
            
            dV_dNdc = torch.stack(spatial_grads, dim=-1)
            dV_dX = torch.matmul(dV_dNdc, sNdc_dX)
            spatial_grads = [sg[...,0] for sg in torch.split(dV_dX, 1, dim=-1)]

            dS_dNdc = torch.stack(spatial_grads_sigma, dim=-1)
            dS_dX = torch.matmul(dS_dNdc[:,None], sNdc_dX)
            spatial_grads_sigma = [sg[...,0,0] for sg in torch.split(dS_dX, 1, dim=-1)]

            

        res['num_dvdx'] = spatial_grads[0] * valid_mask[...,None]
        res['num_dvdy'] = spatial_grads[1] * valid_mask[...,None]
        res['num_dvdz'] = spatial_grads[2] * valid_mask[...,None]
        res['num_dvdt'] = grad_t * valid_mask[...,None]

        res['num_dsigmadx'] = spatial_grads_sigma[0] * valid_mask
        res['num_dsigmady'] = spatial_grads_sigma[1] * valid_mask
        res['num_dsigmadz'] = spatial_grads_sigma[2] * valid_mask
        res['num_dsigmadt'] = grad_t_sigma * valid_mask
        
        res['valid_mask_grad'] = valid_mask
        res['velocity_ref'] *= valid_mask[...,None]
        res['flows_ref'] *= valid_mask[...,None]
        res['sigma'] *= valid_mask

        if 'velocity_ref' in res:
            res['velocity_ref_grad'] = res['velocity_ref'] 
            res['ref_grad_order_1'] = res['velocity_ref']

        if self.flow_order > 1:
            for order in range(2, self.flow_order + 1):
                res['ref_grad_order_%d' % order] = res['flows_ref'][..., 3 * (order-1): 3 * order]

        return res
    
    # run scene-flow bundle
    def compute_sf_bundle(self, xyz_sampled, viewdirs, frame_time, 
                                get_sigma_ref=True, get_rgb_ref=True, get_flow_ref=True, 
                                get_pred_normal=False):

        res = {}
        if get_sigma_ref:
            density_feature = self.compute_densityfeature(
                xyz_sampled, frame_time
            )
            density_feature = self.density_regressor(
                xyz_sampled,
                viewdirs,
                density_feature,
                frame_time,
            )
            validsigma = self.feature2density(density_feature)
            sigma = validsigma.view(-1)

            res['sigma'] = sigma

        if get_rgb_ref or get_pred_normal:
            app_features = self.compute_appfeature(
                xyz_sampled, frame_time
            )
            if not self.use_viewdir_dy:
                viewdirs = vfu.sample_spherical(viewdirs.shape[0])
                viewdirs = viewdirs.to(xyz_sampled.device)

            if get_rgb_ref:
                rgbs = self.app_regressor(
                    xyz_sampled,
                    viewdirs,
                    app_features,
                    frame_time,
                )
                res['rgbs'] = rgbs

            if get_pred_normal:
                resolutions = [self.flow_line_time[2].shape[2],  # resolution of x-axis
                           self.flow_line_time[1].shape[2],  # resolution of y-axis
                           self.flow_line_time[0].shape[2]]  # resolution of z-axis
                
                multiplier = self.nSamples / resolutions[2]
                multiplier *= self.grad_res_multiplier
                if self.align_corners:
                    epsilons = 2. / (np.array(resolutions) * multiplier * 1.0)  # from normalized one (-1 ~ 1)
                else:
                    epsilons = 2. / (np.array(resolutions) * multiplier * 1.0 - 1.) 

                epsilons = torch.FloatTensor(epsilons).to(xyz_sampled.device)

                xyz_sampled_noised = self.normalize_coord(xyz_sampled) + epsilons * (torch.rand_like(xyz_sampled) * 2. - 1.)
                xyz_sampled_noised = torch.clamp(xyz_sampled_noised, -1., 1.)
                xyz_sampled_noised = self.denormalize_coord(xyz_sampled_noised)
                app_features_noised = self.compute_appfeature(
                    xyz_sampled_noised, frame_time
                )
                pred_normal = self.normal_regressor(
                    xyz_sampled_noised,
                    viewdirs,
                    app_features_noised,
                    frame_time,
                )
                res['pred_dsigma'] = pred_normal
                pred_normal = pred_normal / (torch.norm(pred_normal, dim=-1, keepdim=True) + 1e-10)
                res['pred_normal'] = pred_normal

        if get_flow_ref:
            flow_features = self.compute_flowfeature(
                xyz_sampled, frame_time
            )
            flows = self.flow_regressor(
                xyz_sampled,
                viewdirs,
                flow_features,
                frame_time,
            )

            flow_mask = self.taylor_mask
            flow_mask = flow_mask.to(flows.device)
            flow_mask = flow_mask[..., None].repeat(1,3).view(-1)
            scaler = self.aabbSize.to(flows.device).max()
            flows = flows * flow_mask

            sigma_scale = self.sigma_scale

            flow_scaler = torch.ones_like(flows) / self.time_interval

            flows = flows * flow_scaler.detach()
            flows = flows * self.flow_scale / self.time_interval * scaler / sigma_scale

            flows_world = flows
            
            if self.flow_order > 0:
                res['flows_world'] = flows_world  # flows in world coordinates
                res['flows_ref'] = flows_world  # flows in world coordinates
                res['flows_raw'] = flows  # raw flow
                res['velocity_ref'] = flows_world[...,:3]  # velocity should be in world coordinate
            else:
                # when flow_order <= 0, raw flows (output from the flow HexPlane) is used as flows...
                res['flows_ref'] = flows
                res['flows_raw'] = flows
                res['velocity_ref'] = flows[...,:3]
            
        return res
    
    # render from rays_chunk
    def forward(
        self,
        rays_chunk: torch.tensor,
        frame_time: torch.tensor,
        white_bg: bool = True,
        is_train: bool = False,
        ndc_ray: bool = False,
        N_samples: int = -1,
        poses: dict = None,
        render_only: bool = False
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
            poses: dict, poses (prev, current, post) of each ray.
            render_only: bool, render rgb and depth only

        Returns:
            rgb: (B, 3) tensor, rgb values.
            depth: (B, 1) tensor, depth values.
            alpha: (B, 1) tensor, accumulated weights.
            z_vals: (B, N_samples) tensor, z values.
            aux: dict, auxiliary outputs.
        """
        self.white_bg = white_bg

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

        # check empty dynamic case
        no_valid_app = False
        
        if not app_mask.any():
            # exceptional case where there is no dynamic point
            app_mask[0,0] = True  # pseudo input; this will not affect any parameter update.
            no_valid_app = True
            weight = weight.detach()
        
        # store auxiliary elements and will be returned
        aux = {}

        aux['weight_noblend'] = weight_noblend
        aux['stweight_noblend'] = stweight_noblend

        if self.return_xyz:
            aux['xyz_sampled'] = xyz_sampled
            aux['app_mask'] = app_mask
            aux['weight'] = weight
            aux['stweight'] = stweight

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
        
        no_flow = (not self.compute_flow and is_train) or render_only
        
        # compute dynamic rgb and flows
        if app_mask.any():
            # ref coordinates and viewdir
            xyz_app = xyz_sampled[app_mask]
            viewdir_app = viewdirs[app_mask]
            frametime_app = frame_time[app_mask]
            
            if no_flow:
                app_features = self.compute_appfeature(
                    xyz_app, frametime_app, False
                )
                rgbs = self.app_regressor(
                    xyz_app,
                    viewdir_app,
                    app_features,
                    frametime_app,
                )

                if no_valid_app:
                    rgbs = rgbs.detach()

                rgb[app_mask] = rgbs
            else:
                # query at reference time
                bundle_ref = self.compute_sf_bundle(xyz_app, viewdir_app, frametime_app,
                                                    get_sigma_ref=False, # already got sigma
                                                    get_rgb_ref=True, get_pred_normal=True)

                rgb[app_mask] = bundle_ref['rgbs']
                flow_raw_ref = bundle_ref['flows_ref']

                # warped rgb rendering ingredient
                render_args = {'app_mask': app_mask, 'dists': dists, 
                            'stsigma': stsigma, 'stweight': stweight, 'strgb': strgb,
                            'sigma': sigma, 'weight': weight, 'rgb': rgb}
                
                # compute flows for 2D optical flow computation
                flow_ref2post = self.get_flow( 1., xyz_app, flow_raw_ref)
                flow_ref2prev = self.get_flow(-1., xyz_app, flow_raw_ref)
                
                # coordinates at post / prev frametime (discrete)
                xyz_post = self.get_new_coords(xyz_app, flow_ref2post)
                xyz_prev = self.get_new_coords(xyz_app, flow_ref2prev)

            # executed only during training for inference effeciency
            if is_train and not no_flow:
                # random timehop sampling (timehop is only defined on discrete timeframes)
                timehop = 1. + torch.rand(xyz_sampled.shape[0], 
                                          device=xyz_sampled.device) * (self.max_timehop - 1.)  # shape: N_rays (e.g., 4096)
                timehop = torch.floor(timehop)
                timedir = torch.sign(torch.rand(xyz_sampled.shape[0], device=xyz_sampled.device) - 0.5)
                frametime_temp = frame_time[:,0,0] + self.time_interval * timehop * timedir
                time_outbound = (frametime_temp > 1.+1e-5) | (frametime_temp < -1. - 1e-5)
                timedir[time_outbound] = -timedir[time_outbound]
                timehop = timehop * timedir
                timehop_app = timehop[...,None,None].repeat(1, xyz_sampled.shape[1], 1)[app_mask]

                # render from 1 timehop away
                bundle_p, xyz_p, flow_ref2p = self.get_warped_rgb_bundle(xyz_app, viewdir_app, frametime_app,
                                                  timehop_app, flow_raw_ref, aux, render_args, 'p')
                
                # flow timehop -> ref
                flow_p2ref = self.get_flow(-timehop_app, xyz_p, bundle_p['flows_ref'])

                # query inter-frame (with randomized interframe times)
                timeinter = torch.rand(xyz_sampled.shape[0], device=xyz_sampled.device)

                # abs(timeinter) < abs(timehop)
                timeinter = timeinter * torch.minimum(self.max_timeinter + torch.zeros_like(timehop), torch.abs(timehop))

                # direction should be the same
                timeinter *= timedir

                timeinter_app = timeinter[...,None,None].repeat(1, xyz_sampled.shape[1], 1)[app_mask]

                # render from timeinter
                bundle_pinter, xyz_pinter, flow_ref2pinter = self.get_warped_rgb_bundle(xyz_app, viewdir_app, frametime_app,
                                                  timeinter_app, flow_raw_ref, aux, render_args, 'pinter')
                
                frametime_pinter = frametime_app + timeinter_app * self.time_interval

                # store flows
                aux['flow_raw_ref'] = bundle_ref['flows_ref']

                # prevprev <- prev <- ref -> post -> postpost
                # prev_chain <- previnter <- ref -> postinter -> post_chain
                aux['flow_ref2p'] = flow_ref2p
                aux['flow_p2ref'] = flow_p2ref
                aux['flow_ref2pinter'] = flow_ref2pinter

                # compute hessian
                if self.compute_hessian:
                    physics_act = torch.square


                    # sample from rays
                    all_idx = torch.arange(xyz_app.size(0)).to(self.device)
                    perm = torch.randperm(all_idx.size(0))
                    sel_idx = all_idx[perm[0:self.num_hessian_samples]]
                    sample_points = xyz_pinter[sel_idx]
                    sample_viewdirs = viewdir_app[sel_idx] 
                    sample_times = frametime_pinter[sel_idx]
                    sample_weights = weight[app_mask][sel_idx]

                    rigidity_one = 0.
                    rigidity_two = 0.
                    kinematic_integrity = 0.
                    H_pred_mag = 0.
                    H_mag = 0.

                    aabb = torch.FloatTensor(self.aabb).to(sample_points.device)

                    valid_mask_samp = torch.logical_and(sample_points >= aabb[0], sample_points <= aabb[1])
                    valid_mask_samp = torch.all(valid_mask_samp, dim=-1)
                    valid_mask_samp &= (bundle_pinter['sigma'][sel_idx] * self.distance_scale > 1.0)

                    if sample_points.size(0) > 0 and torch.any(valid_mask_samp):
                        bundle_ref_hessian = self.compute_sf_bundle_grad(sample_points[valid_mask_samp], 
                                                                         sample_viewdirs[valid_mask_samp], 
                                                                         sample_times[valid_mask_samp],
                                                                         is_train=is_train,
                                                                         get_sigma_ref=True,
                                                                         get_rgb_ref=False,)

                        pts_grad = sample_points
                            
                        grad_x = bundle_ref_hessian['num_dvdx']
                        grad_y = bundle_ref_hessian['num_dvdy']
                        grad_z = bundle_ref_hessian['num_dvdz']
                        grad_t = bundle_ref_hessian['num_dvdt']

                        grad_x_sigma = bundle_ref_hessian['num_dsigmadx']
                        grad_y_sigma = bundle_ref_hessian['num_dsigmady']
                        grad_z_sigma = bundle_ref_hessian['num_dsigmadz']
                        grad_t_sigma = bundle_ref_hessian['num_dsigmadt']
                            
                        # Divergence compuatation
                        dV_dX = torch.stack([grad_x[...,:3], grad_y[...,:3], grad_z[...,:3]], dim=-1)
                        div = torch.einsum("...ii", dV_dX)
                        rigidity_one += physics_act(div)

                        sc_ = self.densitygrad_scaler

                        # Strain rate computation
                        rigidity_two_tmp = vfu.get_strain_rate_grad(grad_x[...,:3], grad_y[...,:3], grad_z[...,:3], pts_grad, 
                                                                    {'use_ndc':False})
                        rigidity_two += physics_act(rigidity_two_tmp)
                        
                        transport_reg = vfu.compute_higher_field(sc_(grad_x_sigma[...,None]), 
                                                                 sc_(grad_y_sigma[...,None]), 
                                                                 sc_(grad_z_sigma[...,None]), 
                                                                 sc_(grad_t_sigma[...,None]),
                                                                 bundle_ref_hessian['velocity_ref'].detach())
                        
                        transport_reg = physics_act(transport_reg)
                        transport_reg = transport_reg[...,0]
                        
                        # normal
                        normal_grad = torch.cat([-grad_x_sigma[...,None], -grad_y_sigma[...,None], -grad_z_sigma[...,None]], axis=-1)
                        normal_grad_norm = torch.norm(normal_grad, dim=-1, keepdim=True)
                        normal_grad = normal_grad / (normal_grad_norm + 1e-10)
                        valid_grad_mask = (normal_grad_norm > 0.0)[:,0]

                        normal_pred = bundle_ref_hessian['pred_normal']

                        normal_reg = sample_weights[valid_mask_samp][valid_grad_mask][...,None].detach() \
                            * (normal_pred[valid_grad_mask] - normal_grad[valid_grad_mask]) ** 2
                        
                        aux['normal_reg'] = normal_reg

                        vel_grad_sc = self.velocitygrad_scaler

                        # Kinematic integrity loss
                        for i in range(1, self.flow_order):
                            L = bundle_ref_hessian["ref_grad_order_%d" % i]
                            H = bundle_ref_hessian["ref_grad_order_%d" % (i+1)]

                            grad_x_, grad_y_, grad_z_, grad_t_ = (grad_x[...,(i-1)*3:i*3], 
                                                                  grad_y[...,(i-1)*3:i*3], 
                                                                  grad_z[...,(i-1)*3:i*3],
                                                                  grad_t[...,(i-1)*3:i*3])

                            H_pred = vfu.compute_higher_field(grad_x_, grad_y_, grad_z_, grad_t_,
                                                              bundle_ref_hessian['velocity_ref'])
                            
                            H_pred = vel_grad_sc(H_pred)
                            H_pred = torch.clamp(H_pred, -64., 64.)

                            H = H * self.taylor_mask[i]
                            H_pred = H_pred * self.taylor_mask[i]

                            H_pred_mag += torch.mean((H_pred) ** 2, dim=-1)
                            H_mag += torch.mean((H) ** 2, dim=-1)

                            diff = torch.mean((H - H_pred) ** 2, dim=-1)
                            kinematic_integrity += diff
                    
                        # normalize time-interval-dependent losses
                        aux['rigidity_one'] = rigidity_one * (self.time_interval ** 2)
                        aux['rigidity_two'] = rigidity_two * (self.time_interval ** 2)
                        aux['kinematic_integrity'] = kinematic_integrity * (self.time_interval ** 2)
                        aux['transport_reg'] = transport_reg * (self.time_interval ** 2)
                        aux['H_pred_mag'] = H_pred_mag * (self.time_interval ** 2)
                        aux['H_mag'] = H_mag * (self.time_interval ** 2)
        
            if is_train and not no_flow:
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
                    
            elif not no_flow:
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
            # for velocity field rendering
            if self.render_kinematic_field and self.flow_order > 1:
                with torch.no_grad():
                    order_names = ["velocity", "acceleration", "jerk", "snap", "crackle", "pop"]
                    for order in range(1, self.flow_order + 1):
                        val = torch.zeros_like(flow_raw_ref)
                        val[...,(order-1) * 3:order * 3] = flow_raw_ref[...,(order-1) * 3:order * 3]

                        val_only = self.get_new_coords(xyz_app, self.get_flow(1., xyz_app, val))

                        aux['%s_map_fw' %  order_names[order - 1]] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                        val_only, poses['pose'], poses['pose'], weight, stweight)

                        
                    bundle_ref_hessian = self.compute_sf_bundle_grad(xyz_app, 
                                                                        viewdir_app, 
                                                                        frametime_app,
                                                                        get_sigma_ref=True,
                                                                        get_rgb_ref=False,)
                    
                    grad_x = bundle_ref_hessian['num_dvdx']
                    grad_y = bundle_ref_hessian['num_dvdy']
                    grad_z = bundle_ref_hessian['num_dvdz']
                    grad_t = bundle_ref_hessian['num_dvdt']

                    Hs, H_preds = [], []
                    for i in range(1, self.flow_order):
                        grad_x_, grad_y_, grad_z_, grad_t_ = (grad_x[...,(i-1)*3:i*3], 
                                                                grad_y[...,(i-1)*3:i*3], 
                                                                grad_z[...,(i-1)*3:i*3],
                                                                grad_t[...,(i-1)*3:i*3])

                        H_pred = vfu.compute_higher_field(grad_x_, grad_y_, grad_z_, grad_t_,
                                                            bundle_ref_hessian['velocity_ref'])
                        H_preds.append(H_pred)

                    for order in range(2, self.flow_order + 1):
                        val = torch.zeros_like(flow_raw_ref)
                        val[...,(order-1) * 3:order * 3] = H_preds[order - 2]

                        val_only_pred = self.get_new_coords(xyz_app, self.get_flow(1., xyz_app, val))
                        aux['%s_map_fw_pred' % order_names[order - 1]] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                            val_only_pred, poses['pose'], poses['pose'], weight, stweight)


        # if there was no valid sigmas in the dynamic field, detach all outputs
        if no_valid_app:
            rgb = rgb.detach()
            weight_noblend = weight_noblend.detach()
            rgb_blended = rgb_blended.detach()
            aux = {k: v.detach() for k, v in aux.items()}


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
            dydepth_map = dydepth_map / acc_map.clip(1e-12)

            stdepth_map = torch.sum(stweight_noblend * z_vals, -1)
            stdepth_map = stdepth_map / stacc_map.clip(1e-12)

            aux['dydepth_map'] = dydepth_map
            aux['stdepth_map'] = stdepth_map

            depth_map_blended = torch.sum(weight_blended * z_vals, -1)
            depth_map_blended = depth_map_blended / acc_map_blended.clip(1e-12)
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
    

    def normalize_deriv(self, deriv):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """

        num_repeat = deriv.shape[-1] // 3
        if self.normalize_type == "normal":
            aabb = torch.FloatTensor(self.aabb).to(self.device)
            invaabbSize = self.invaabbSize.clone().to(deriv.device)

            repeat_param = [1] * (len(aabb.shape) - 1) + [num_repeat]
            aabb0 = aabb[0].repeat(repeat_param)
            invaabbSize = invaabbSize.repeat(repeat_param)
            
            return deriv * invaabbSize

    def denormalize_deriv(self, deriv):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """

        num_repeat = deriv.shape[-1] // 3
        if self.normalize_type == "normal":
            aabb = torch.FloatTensor(self.aabb).to(self.device)
            invaabbSize = self.invaabbSize.clone().to(deriv.device)

            repeat_param = [1] * (len(aabb.shape) - 1) + [num_repeat]
            aabb0 = aabb[0].repeat(repeat_param)
            invaabbSize = invaabbSize.repeat(repeat_param)


            return deriv / invaabbSize
        
    def get_warped_rgb_blend(self, app_mask, dists, 
                             stsigma, stweight, strgb, 
                             dysigma_app, dyrgb_app, refsigma, refweight, refrgb):
        stsigma = stsigma.detach()
        strgb = strgb.detach()
        
        dists_scaled = dists * self.distance_scale

        dyrgb = torch.zeros_like(refrgb)
        dyrgb[app_mask] = dyrgb_app

        dysigma = torch.zeros_like(refsigma)
        dysigma[app_mask] = dysigma_app

        alpha = 1.0 - torch.exp(-dysigma * dists_scaled)
        stalpha = 1.0 - torch.exp(-stsigma * dists_scaled)

        alpha_blended = 1.0 - torch.exp(-(dysigma + stsigma) * dists_scaled)
        T_blended = torch.cumprod(
            torch.cat(
                [torch.ones(alpha.shape[0], 1).to(alpha.device), (1.0 - alpha_blended) + 1e-10], -1
            ),
            -1,
        )

        T_noblend = torch.cumprod(
            torch.cat(
                [torch.ones(alpha.shape[0], 1).to(alpha.device), (1.0 - alpha) + 1e-10], -1
            ),
            -1,
        )
        
        weight = alpha * T_blended[...,:-1]
        weight_noblend = alpha * T_noblend[...,:-1]

        stweight_warped = stalpha * T_blended[...,:-1]
        acc_map_blended = torch.sum(weight + stweight_warped, -1)

        rgb_blended = dyrgb * weight[...,None] + \
                      strgb.detach() * stweight_warped[...,None]
        rgb_map_blended = torch.sum(rgb_blended, -2)
        

        if self.white_bg:
            rgb_map_blended = rgb_map_blended + (1.0 - acc_map_blended[..., None])
        else:
            rgb_map_blended = rgb_map_blended + (1.0 - acc_map_blended[..., None]) * torch.rand(
                size=(1, 3), device=rgb_map_blended.device
            )
        
        dy_prob = (refsigma / (refsigma + stsigma + 1e-16)).clone().detach()
        dy_prob_map = torch.sum(dy_prob * refweight, -1)
        dy_prob_map = dy_prob_map.detach()
        dy_prob_sampled = dy_prob[app_mask]

        rgb_map_blended = rgb_map_blended.clamp(0, 1)

        return rgb_map_blended, dy_prob_sampled, dy_prob_map, weight, weight_noblend
    

    def get_new_coords(self, xyz, flow):
        if self.ndc_system:
            xyz_moved = vfu.get_new_coords(xyz, flow, self.ndc_params, epsilon=1e-9)
            xyz_moved = self.clamp_pts(xyz_moved, epsilon=1e-9)
        else:
            xyz_moved = xyz + flow
            xyz_moved = self.clamp_pts(xyz_moved, epsilon=1e-9)

        return xyz_moved

    def get_flow(self, t, xyz, flow_raw):
        t = t * self.time_interval

        flow = self.get_disp_at(t, flow_raw) 
        
        return flow

    def get_disp_at(self, t, flow_raw):
        if self.flow_order > 0:
            return vfu.get_disp_at(t, self.flow_order, flow_raw)
        else:
            t = t + torch.zeros_like(flow_raw[...,:1])
            fw_mask = (t > 0)[...,0]
            bw_mask = torch.logical_not(fw_mask)

            flow = t * flow_raw[...,3:]
            flow[bw_mask] = (flow_raw[...,:3] * (-t))[bw_mask] 
            return flow
    

    def compute_sf_sm(self, xyz, app_mask, xyz_targ):
        xyz_all = xyz.clone().detach()
        xyz_all[app_mask] = xyz_targ

        if self.ndc_system:
            xyz_world = vfu.ndc2world(xyz, **self.ndc_params)
            xyz_targ_world = vfu.ndc2world(xyz_all, **self.ndc_params)
        else:
            xyz_world = xyz
            xyz_targ_world = xyz_all

        n = xyz_world.shape[-2]

        xyz_world = xyz_world[..., :int(n * 0.95), :]
        xyz_targ_world = xyz_targ_world[..., :int(n * 0.95), :]

        sm = vfu.compute_sf_sm(xyz_targ_world - xyz_world)
        return sm
    
    def compute_normal_ndc(self, xyz_app, normal_app):
        xyz_world = vfu.ndc2world(xyz_app, **self.ndc_params)
        xyz_targ_world = xyz_world + normal_app
        xyz_targ = vfu.world2ndc(xyz_targ_world, **self.ndc_params)
        normal_ndc = xyz_targ - xyz_app
        normal_ndc = normal_ndc / torch.norm(normal_ndc, dim=-1, keepdim=True)
        return normal_ndc

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        flow_res_target = [int(targ * self.num_grad_res_scale) for targ in res_target]
        flow_time_grid = int(time_grid * self.num_grad_res_scale) 
        self.flow_plane, self.flow_line_time = self.up_sampling_planes(
            self.flow_plane, self.flow_line_time, flow_res_target, flow_time_grid
        )

        # perform only if the static model has lower resolution
        if self.static_model.density_plane[0].shape[-1] < self.density_plane[0].shape[-1]:
            self.static_model.upsample_volume_grid(res_target, time_grid)

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")
        print("plane sizes:")
        print("  app_plane: ")
        for p, v in zip(self.app_plane, self.app_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  density_plane: ")
        for p, v in zip(self.density_plane, self.density_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  stapp_plane: ")
        for p, v in zip(self.static_model.app_plane, self.static_model.app_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  stdensity_plane: ")
        for p, v in zip(self.static_model.density_plane, self.static_model.density_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  flow_plane: ")
        for p, v in zip(self.flow_plane, self.flow_line_time):
            print("  - %s, %s" % (p.shape, v.shape))

    def TV_loss_flow(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.flow_plane)):
            total = total + reg(self.flow_plane[idx]) + reg2(self.flow_line_time[idx])
        return total
    
    def render_optical_flow(self, xyz, app_mask, xyz_targ, pose_ref, pose_targ, dyweight, stweight):
        xyz_all = xyz.clone().detach()
        xyz_all[app_mask] = xyz_targ
        ndc_params = self.ndc_params
        
        if self.ndc_system:
            optflow = vfu.compute_optical_flow(pose_ref, pose_targ, 
                                           ndc_params['H'], ndc_params['W'], ndc_params['focal'],
                                           dyweight, stweight, xyz, 
                                           xyz_all, center=ndc_params['center'])
        else:
            optflow = vfu.compute_optical_flow_world(pose_ref, pose_targ,
                                            ndc_params['H'], ndc_params['W'], ndc_params['focal'],
                                            dyweight, stweight, xyz, 
                                            xyz_all, center=ndc_params['center'])
        optflow[torch.isnan(optflow)] = 0.
        optflow = torch.clamp(optflow, -400., 400.)
        return optflow
    
    def add_one_taylor_order(self):
        if self.use_taylor_order_mask:
            taylor_mask = self.taylor_mask
            for i, value in enumerate(taylor_mask):
                if value == 0.:
                    taylor_mask[i] = 1.0
                    break


    def get_warped_rgb_bundle(self, xyz, viewdir, frametime, 
                       time_offset, flow_raw, aux, render_args, append):
        ra = render_args
        flow = self.get_flow(time_offset, xyz, flow_raw)
        xyz_moved = self.get_new_coords(xyz, flow)
        xyz_moved = self.clamp_pts(xyz_moved)

        xyz_moved = self.flowgrad_scaler(xyz_moved)

        bundle = self.compute_sf_bundle(xyz_moved, viewdir, 
                                        frametime + time_offset * self.time_interval,
                                        get_sigma_ref=True,
                                        get_rgb_ref=True)


        dyrgb = bundle['rgbs']
        dysigma = bundle['sigma']

        rgb_warped, dy_prob, dy_prob_map, weight_warped, weight_noblend = self.get_warped_rgb_blend(ra['app_mask'], ra['dists'], ra['stsigma'], ra['stweight'], ra['strgb'], 
                                                                dysigma, dyrgb, ra['sigma'], ra['weight'], ra['rgb'])
        

        aux['rgb_warped_' + append] = rgb_warped
        aux['dy_prob_' + append] = dy_prob
        aux['dy_mask_' + append] = dy_prob_map
        aux['weight_' + append] = weight_warped
        aux['weight_noblend_' + append] = weight_noblend

        return bundle, xyz_moved, flow

    def valid_pts_mask(self, pts):
        aabb = torch.tensor(self.aabb).to(pts.device)

        mask_outbbox = ((aabb[0] > pts) | (pts > aabb[1])).any(
            dim=-1
        )

        return ~mask_outbbox

    def clamp_pts(self, pts, epsilon=0.0):
        aabb = torch.tensor(self.aabb).to(pts.device)

        pts_x = torch.clamp(pts[...,0], aabb[0][0] + epsilon, aabb[1][0] - epsilon)
        pts_y = torch.clamp(pts[...,1], aabb[0][1] + epsilon, aabb[1][1] - epsilon)
        pts_z = torch.clamp(pts[...,2], aabb[0][2] + epsilon, aabb[1][2] - epsilon)

        res = torch.stack([pts_x, pts_y, pts_z], dim=-1)
        return res

    def photo_loss(self, warped_rgb, gt_rgb, mask):
        loss = torch.mean(mask[...,None].detach() * vfu.chabonnier(warped_rgb - gt_rgb))
        return loss