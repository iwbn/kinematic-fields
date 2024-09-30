import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
from torch.nn import functional as F

from hexplane.model.mlp import General_MLP
from hexplane.model.sh import eval_sh_bases

import hexplane.model.vf_utils as wru

def raw2alpha(sigma: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    rgb = features
    return rgb


def DensityRender(
    xyz_sampled: torch.Tensor,
    viewdirs: torch.Tensor,
    features: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    density = features
    return density


class EmptyGridMask(torch.nn.Module):
    def __init__(
        self, device: torch.device, aabb: torch.Tensor, empty_volume: torch.Tensor
    ):
        super().__init__()
        self.device = device

        self.aabb = aabb #.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        empty_volume = empty_volume.view(1, 1, *empty_volume.shape[-3:])
        self.register_buffer('empty_volume', empty_volume) # to include this in the state_dict()
        self.gridSize = torch.LongTensor(
            [empty_volume.shape[-1], empty_volume.shape[-2], empty_volume.shape[-3]]
        ).to(self.device)

    def sample_empty(self, xyz_sampled):
        empty_volume = self.empty_volume.to(xyz_sampled.device)
        empty_vals = F.grid_sample(
            empty_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)
        return empty_vals


class TensoRF_Slim(torch.nn.Module):
    """
    HexPlane Base Class.
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        gridSize: List[int],
        device: torch.device,
        time_grid: int,
        near_far: List[float],
        stdensity_n_comp: Union[int, List[int]] = 8,
        stapp_n_comp: Union[int, List[int]] = 24,
        stdensity_dim: int = 1,
        stapp_dim: int = 27,
        StDensityMode: str = "plain",
        StAppMode: str = "general_MLP",
        emptyMask: Optional[EmptyGridMask] = None,
        fusion_one: str = "multiply",
        fusion_two: str = "concat",
        fea2denseAct: str = "softplus",
        init_scale: float = 0.1,
        init_shift: float = 0.0,
        normalize_type: str = "normal",
        **kwargs,
    ):
        super().__init__()

        self.aabb = aabb
        self.device = device
        self.near_far = near_far
        self.near_far_org = near_far
        self.step_ratio = kwargs.get("step_ratio", 2.0)
        self.update_stepSize(gridSize)

        # Density and Appearance HexPlane components numbers and value regression mode.
        self.density_n_comp = stdensity_n_comp
        self.app_n_comp = stapp_n_comp
        self.density_dim = stdensity_dim
        self.app_dim = stapp_dim
        self.align_corners = kwargs.get(
            "align_corners", True
        )  # align_corners for grid_sample

        # HexPlane weights initialization: scale and shift for uniform distribution.
        self.init_scale = init_scale
        self.init_shift = init_shift

        # for ndc <-> world conversion
        self.ndc_system = kwargs.get("ndc_system", False)
        self.ndc_params = kwargs['ndc_params']

        # HexPlane fusion mode.
        self.fusion_one = fusion_one
        self.fusion_two = fusion_two

        # Plane Index
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

        # Coordinate normalization type.
        self.normalize_type = normalize_type

        # Plane initialization.
        self.init_planes(gridSize[0], device)

        # Density calculation settings.
        self.fea2denseAct = fea2denseAct  # feature to density activation function
        self.density_shift = kwargs.get(
            "density_shift", -10.0
        )  # density shift for density activation function.
        self.distance_scale = kwargs.get(
            "distance_scale", 25.0
        )  # distance scale for density activation function.

        self.DensityMode = StDensityMode
        self.density_t_pe = kwargs.get("stdensity_t_pe", -1)
        self.density_pos_pe = kwargs.get("stdensity_pos_pe", -1)
        self.density_view_pe = kwargs.get("stdensity_view_pe", -1)
        self.density_fea_pe = kwargs.get("stdensity_fea_pe", 6)
        self.density_featureC = kwargs.get("stdensity_featureC", 128)
        self.density_n_layers = kwargs.get("stdensity_n_layers", 3)
        self.init_density_func(
            self.DensityMode,
            self.density_t_pe,
            self.density_pos_pe,
            self.density_view_pe,
            self.density_fea_pe,
            self.density_featureC,
            self.density_n_layers,
            self.device,
        )

        # Appearance calculation settings.
        self.AppMode = StAppMode
        self.app_t_pe = kwargs.get("stapp_t_pe", -1)
        self.app_pos_pe = kwargs.get("stapp_pos_pe", -1)
        self.app_view_pe = kwargs.get("stapp_view_pe", 6)
        self.app_fea_pe = kwargs.get("stapp_fea_pe", 6)
        self.app_featureC = kwargs.get("stapp_featureC", 128)
        self.app_n_layers = kwargs.get("stapp_n_layers", 3)
        self.init_app_func(
            StAppMode,
            self.app_t_pe,
            self.app_pos_pe,
            self.app_view_pe,
            self.app_fea_pe,
            self.app_featureC,
            self.app_n_layers,
            device,
        )

        # Density HexPlane mask and other acceleration tricks.
        self.emptyMask = emptyMask
        self.emptyMask_thres = kwargs.get(
            "emptyMask_thres", 0.001
        )  # density threshold for emptiness mask
        self.rayMarch_weight_thres = kwargs.get(
            "rayMarch_weight_thres", 0.0001
        )  # density threshold for rendering colors.

        # Regulartization settings.
        self.random_background = kwargs.get("random_background", False)
        self.depth_loss = kwargs.get("depth_loss", False)



    def init_density_func(
        self, DensityMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device
    ):
        """
        Initialize density regression function.
        """
        if (
            DensityMode == "plain"
        ):  # Use extracted features directly from HexPlane as density.
            assert self.density_dim == 1  # Assert the extracted features are scalers.
            self.density_regressor = DensityRender
        elif DensityMode == "general_MLP":  # Use general MLP to regress density.
            assert (
                view_pe < 0
            )  # Assert no view position encoding. Density should not depend on view direction.
            self.density_regressor = General_MLP(
                self.density_dim,
                1,
                -1,
                fea_pe,
                pos_pe,
                view_pe,
                featureC,
                n_layers,
                use_sigmoid=False,
                zero_init=False,
            ).to(device)
        else:
            raise NotImplementedError("No such Density Regression Mode")
        print("STATIC DENSITY REGRESSOR:")
        print(self.density_regressor)

    def init_app_func(
        self, AppMode, t_pe, pos_pe, view_pe, fea_pe, featureC, n_layers, device
    ):
        """
        Initialize appearance regression function.
        """
        if AppMode == "SH":  # Use Spherical Harmonics SH to render appearance.
            self.app_regressor = SHRender
        elif AppMode == "RGB":  # Use RGB to render appearance.
            assert self.app_dim == 3
            self.app_regressor = RGBRender
        elif AppMode == "general_MLP":  # Use general MLP to render appearance.
            self.app_regressor = General_MLP(
                self.app_dim,
                3,
                -1,
                fea_pe,
                pos_pe,
                view_pe,
                featureC,
                n_layers,
                use_sigmoid=True,
                zero_init=True,
            ).to(device)
        else:
            raise NotImplementedError("No such App Regression Mode")
        print("STAPP REGRESSOR:")
        print(self.app_regressor)
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)

    # initialize each plane. Note that flow-related "dy_plane" and "flow_plane" are added.
    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        self.app_plane, self.app_line_time = self.init_one_tensorf(
            self.app_n_comp, self.gridSize, device
        )

        self.density_plane, self.density_line_time = self.init_one_tensorf(
            self.density_n_comp, self.gridSize, device
        )

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

        # Initialize the basis matrices
        with torch.no_grad():
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            self.density_basis_mat.weight.copy_(weights)
        
        print("plane sizes:")
        print("  static_app_plane: ")
        for p, v in zip(self.app_plane, self.app_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
        print("  static_density_plane: ")
        for p, v in zip(self.density_plane, self.density_line_time):
            print("  - %s, %s" % (p.shape, v.shape))
    
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
                (torch.zeros_like(line_time_coord), line_time_coord), dim=-1
            )
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        # Extract features from six feature planes.
        for idx_plane in range(len(self.density_plane)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.density_line_time[idx_plane],
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
                (torch.zeros_like(line_time_coord), line_time_coord), dim=-1
            )
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.app_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.app_line_time[idx_plane],
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

        inter = self.app_basis_mat(inter.T)  # Feature Projection

        return inter

    def init_one_tensorf(self, n_component, gridSize, device):
        plane_coef, line_time_coef, line_coef = [], [], []

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
            line_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                    + self.init_shift
                )
            )

        return (torch.nn.ParameterList(plane_coef).to(device), 
                torch.nn.ParameterList(line_coef).to(device),
        )

    # will be used to set trainable variables.
    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_stdensity_grid,
                "lr_org": cfg.lr_stdensity_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_stdensity_grid,
                "lr_org": cfg.lr_stdensity_grid,
            },
            {
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_stapp_grid,
                "lr_org": cfg.lr_stapp_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_stapp_grid,
                "lr_org": cfg.lr_stapp_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_stdensity_nn,
                "lr_org": cfg.lr_stdensity_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_stapp_nn,
                "lr_org": cfg.lr_stapp_nn,
            },
        ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_stapp_nn,
                    "lr_org": cfg.lr_stapp_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_stdensity_nn,
                    "lr_org": cfg.lr_stdensity_nn,
                }
            ]
        
        return grad_vars

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.reshape(-1))
        print("grid size", gridSize)
        self.aabbSize = torch.FloatTensor(self.aabb[1] - self.aabb[0]).to(self.device)
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def normalize_coord(self, xyz_sampled):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """

        if self.normalize_type == "normal":
            aabb = torch.FloatTensor(self.aabb).to(xyz_sampled.device)
            invaabbSize = self.invaabbSize.clone().to(xyz_sampled.device)
            return (xyz_sampled - aabb[0]) * invaabbSize - 1
        
    def denormalize_coord(self, xyz_sampled):
        """
        Normalize the sampled coordinates to [-1, 1] range.
        """

        if self.normalize_type == "normal":
            aabb = torch.FloatTensor(self.aabb).to(self.device)
            invaabbSize = self.invaabbSize.clone().to(self.device)
            return (xyz_sampled + 1.) / invaabbSize + aabb[0]

    def feature2density(self, density_features: torch.tensor) -> torch.tensor:
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "gelu":
            MIN_VAL_GELU = -0.1700407506
            return F.gelu(density_features, approximate="tanh") - MIN_VAL_GELU
        else:
            raise NotImplementedError("No such activation function for density feature")

    def sample_rays(
        self,
        rays_o: torch.tensor,
        rays_d: torch.tensor,
        is_train: bool = True,
        N_samples: int = -1,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample points along rays based on the given ray origin and direction.

        Args:
            rays_o: (B, 3) tensor, ray origin.
            rays_d: (B, 3) tensor, ray direction.
            is_train: bool, whether in training mode.
            N_samples: int, number of samples along each ray.

        Returns:
            rays_pts: (B, N_samples, 3) tensor, sampled points along each ray.
            interpx: (B, N_samples) tensor, sampled points' distance to ray origin.
            ~mask_outbbox: (B, N_samples) tensor, mask for points within bounding box.
        """
        aabb = torch.tensor(self.aabb).to(rays_o.device)
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)
            interpx = interpx.clamp(near, far - 1e-10)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def filtering_rays(
        self,
        all_rays: torch.tensor,
        all_rgbs: torch.tensor,
        all_times: torch.tensor,
        all_depths: Optional[torch.tensor] = None,
        N_samples: int = 256,
        chunk: int = 10240 * 5,
        bbox_only: bool = False,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, Optional[torch.tensor]]:
        """
        Filter out rays that are not within the bounding box.

        Args:
            all_rays: (N_rays, N_samples, 6) tensor, rays [rays_o, rays_d].
            all_rgbs: (N_rays, N_samples, 3) tensor, rgb values.
            all_times: (N_rays, N_samples) tensor, time values.
            all_depths: (N_rays, N_samples) tensor, depth values.
            N_samples: int, number of samples along each ray.

        Returns:
            all_rays: (N_rays, N_samples, 6) tensor, filtered rays [rays_o, rays_d].
            all_rgbs: (N_rays, N_samples, 3) tensor, filtered rgb values.
            all_times: (N_rays, N_samples) tensor, filtered time values.
            all_depths: Optional, (N_rays, N_samples) tensor, filtered depth values.
        """
        print("========> filtering rays ...")
        tt = time.time()
        N = torch.tensor(all_rays.shape[:-1]).prod()
        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            # Filter based on bounding box.
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(
                    -1
                )  # clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(
                    -1
                )  # clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            # Filter based on emptiness mask.
            else:
                xyz_sampled, _, _ = self.sample_ray(
                    rays_o, rays_d, N_samples=N_samples, is_train=False
                )
                xyz_sampled = self.normalize_coord(xyz_sampled)
                mask_inbbox = (
                    self.emptyMask.sample_empty(xyz_sampled).view(
                        xyz_sampled.shape[:-1]
                    )
                    > 0
                ).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(
            f"Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}"
        )
        if all_depths is not None:
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_times[mask_filtered],
                all_depths[mask_filtered],
            )
        else:
            return (
                all_rays[mask_filtered],
                all_rgbs[mask_filtered],
                all_times[mask_filtered],
                None,
            )

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
        dists = torch.cat(
            (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1
        )
        rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
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
        alpha, weight, bg_weight = raw2alpha(
            sigma, dists * self.distance_scale
        )  # alpha is the opacity, weight is the accumulated weight. bg_weight is the accumulated weight for last sampling point.

        # Compute appearance feature and rgb if there are valid rays (whose weight are above a threshold).
        app_mask = weight > self.rayMarch_weight_thres
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
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        aux = {}
        if is_train:
            
            xyz_post = xyz_sampled[app_mask]
            xyz_prev = xyz_sampled[app_mask]
            
            # render optical flow (forward) at the next pose
            if poses is not None and "post_pose" in poses.keys():
                aux['optical_flow_fw_post'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_post, poses['pose'], poses['post_pose'], torch.zeros_like(weight), weight)

            # render optical flow (backward) at the prev pose 
            if poses is not None and "prev_pose" in poses.keys():
                aux['optical_flow_bw_prev'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_prev, poses['pose'], poses['prev_pose'], torch.zeros_like(weight), weight)

            # render optical flow (forward and backward) at the ref pose
            if poses is not None and "pose" in poses.keys():
                aux['optical_flow_fw_ref'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_post, poses['pose'], poses['pose'], torch.zeros_like(weight), weight)
                aux['optical_flow_bw_ref'] = self.render_optical_flow(xyz_sampled, app_mask, 
                                                                                xyz_prev, poses['pose'], poses['pose'], torch.zeros_like(weight), weight)

        # If white_bg or (is_train and torch.rand((1,))<0.5):
        if white_bg or not is_train:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        else:
            rgb_map = rgb_map + (1.0 - acc_map[..., None]) * torch.rand(
                size=(1, 3), device=rgb_map.device
            )
        rgb_map = rgb_map.clamp(0, 1)

        # Calculate depth.
        if self.depth_loss:
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        else:
            with torch.no_grad():
                depth_map = torch.sum(weight * z_vals, -1)
                depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        return rgb_map, depth_map, alpha, z_vals, aux

    @torch.no_grad()
    def updateEmptyMask(self, gridSize=(200, 200, 200), time_grid=64):
        """
        Like TensoRF, we compute the emptiness voxel to store the opacities of the scene and skip computing the opacities of rays with low opacities.
        For HexPlane, the emptiness voxel is the union of the density volumes of all the frames.

        This is the same idea as AlphaMask in TensoRF, while we rename it for better understanding.

        Note that we always assume the voxel is a cube [-1, 1], and we sample for normalized coordinate.
        TODO: add voxel shrink function and inverse normalization functions.
        """
        emptiness, dense_xyz = self.getDenseEmpty(gridSize, time_grid)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        emptiness = emptiness.clamp(0, 1).transpose(0, 2).contiguous()[None, None]

        ks = 3
        emptiness = F.max_pool3d(
            emptiness, kernel_size=ks, padding=ks // 2, stride=1
        ).view(gridSize[::-1])
        emptiness[emptiness >= self.emptyMask_thres] = 1
        emptiness[emptiness < self.emptyMask_thres] = 0

        self.emptyMask = EmptyGridMask(self.device, self.aabb, emptiness)

        return None

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
            time_samples = torch.linspace(-1, 1, time_grid).to(xyz_sampled.device)
            N, T = xyz_sampled.shape[0], time_samples.shape[0]
            xyz_sampled = (
                xyz_sampled.unsqueeze(1).expand(-1, T, -1).contiguous().view(-1, 3)
            )
            time_samples = (
                time_samples.unsqueeze(0).expand(N, -1).contiguous().view(-1, 1)
            )

            density_feature = self.compute_densityfeature(xyz_sampled, time_samples)
            sigma_feature = self.density_regressor(
                xyz_sampled,
                xyz_sampled,
                density_feature,
                time_samples,
            ).view(N, T)

            sigma_feature = torch.amax(sigma_feature, -1)
            validsigma = self.feature2density(sigma_feature)
            sigma[empty_mask] = validsigma

        emptiness = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return emptiness

    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx])
            )
        return total

    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx])
        return total
    
    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.app_plane, self.app_line_time = self.up_sampling_planeline(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        self.density_plane, self.density_line_time = self.up_sampling_planeline(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")
    
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

    def sample_rays(
        self,
        rays_o: torch.tensor,
        rays_d: torch.tensor,
        is_train: bool = True,
        N_samples: int = -1,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample points along rays based on the given ray origin and direction.

        Args:
            rays_o: (B, 3) tensor, ray origin.
            rays_d: (B, 3) tensor, ray direction.
            is_train: bool, whether in training mode.
            N_samples: int, number of samples along each ray.

        Returns:
            rays_pts: (B, N_samples, 3) tensor, sampled points along each ray.
            interpx: (B, N_samples) tensor, sampled points' distance to ray origin.
            ~mask_outbbox: (B, N_samples) tensor, mask for points within bounding box.
        """
        aabb = torch.tensor(self.aabb).to(rays_o.device)
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)
            interpx = interpx.clamp(near, far - 1e-10)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(
            dim=-1
        )
        return rays_pts, interpx, ~mask_outbbox

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