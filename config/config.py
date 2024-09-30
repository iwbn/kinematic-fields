from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class System_Config:
    seed: int = 20220401
    basedir: str = "./log"
    ckpt: Optional[str] = None
    progress_refresh_rate: int = 10
    vis_every: int = 10000
    add_timestamp: bool = True


@dataclass
class Model_Config:
    model_name: str = "HexPlaneSD_Flow"  # choose from "HexPlane", "HexPlaneSD_Flow"
    N_voxel_init: int = 64 * 64 * 64  # initial voxel number
    N_voxel_final: int = 200 * 200 * 200  # final voxel number
    step_ratio: float = 0.5
    nonsquare_voxel: bool = True  # if yes, voxel numbers along each axis depend on scene length along each axis
    time_grid_init: int = 16
    time_grid_final: int = 128
    normalize_type: str = "normal"
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    update_emptymask_list: List[int] = field(
        default_factory=lambda: [4000, 8000, 10000]
    )

    # Plane Initialization
    density_n_comp: List[int] = field(default_factory=lambda: [24, 24, 24])
    app_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])
    flow_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])
    stdensity_n_comp: List[int] = field(default_factory=lambda: [24, 24, 24])
    stapp_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])

    density_dim: int = 1
    app_dim: int = 27
    stdensity_dim: int = 1
    stapp_dim: int = 27
    
    DensityMode: str = "plain"  # choose from "plain", "general_MLP"
    AppMode: str = "general_MLP"
    FlowMode: str = "general_MLP"
    StDensityMode: str = "plain"  # choose from "plain", "general_MLP"
    StAppMode: str = "general_MLP"
    
    init_scale: float = 0.1
    init_shift: float = 0.0

    # Fusion Methods
    fusion_one: str = "multiply"
    fusion_two: str = "concat"

    # Density Feature Settings
    fea2denseAct: str = "relu"
    density_shift: float = -10.0
    distance_scale: float = 25.0

    # Density Regressor MLP settings
    density_t_pe: int = -1
    density_pos_pe: int = -1
    density_view_pe: int = -1
    density_fea_pe: int = 2
    density_featureC: int = 128
    density_n_layers: int = 3

    # Static Density Regressor MLP settings
    stdensity_t_pe: int = -1
    stdensity_pos_pe: int = -1
    stdensity_view_pe: int = -1
    stdensity_fea_pe: int = 2
    stdensity_featureC: int = 128
    stdensity_n_layers: int = 3

    # Appearance Regressor MLP settings
    app_t_pe: int = -1
    app_pos_pe: int = -1
    app_view_pe: int = 0
    app_fea_pe: int = 0
    app_featureC: int = 128
    app_n_layers: int = 3
    app_view_pe_begin: int = 0

    # Appearance Regressor MLP settings
    stapp_t_pe: int = -1
    stapp_pos_pe: int = -1
    stapp_view_pe: int = 0
    stapp_fea_pe: int = 0
    stapp_featureC: int = 128
    stapp_n_layers: int = 3

    # Flow Regressor MLP settings
    flow_t_pe: int = -1
    flow_pos_pe: int = -1
    flow_view_pe: int = -1
    flow_fea_pe: int = 0
    flow_featureC: int = 128
    flow_n_layers: int = 3
    flow_order: int = 3

    # Empty mask settings
    emptyMask_thes: float = 0.001
    rayMarch_weight_thres: float = 0.0001

    # Reg
    random_background: bool = False
    depth_loss: bool = False
    depth_loss_weight: float = 1.0
    dist_loss: bool = False
    dist_loss_begin: int = 0
    dist_loss_weight: float = 0.0
    stdist_loss_weight: float = 0.0

    TV_t_s_ratio: float = 64.0  # ratio of TV loss along temporal and spatial dimensions
    TV_weight_density: float = 0.001
    TV_weight_app: float = 0.001
    TV_weight_flow: float = 0.0
    L1_weight_density: float = 0.0
    L1_weight_app: float = 0.0
    L1_weight_flow: float = 0.0

    TV_weight_stapp: float = 0.001
    TV_weight_stdensity: float = 0.001

    L1_weight_stdensity: float = 0.0
    L1_weight_stapp: float = 0.0

    # Sampling
    align_corners: bool = True
    # There are two types of upsampling: aligned and unaligned.
    # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    # using "unaligned" upsampling will essentially double the grid sizes at each time, ignoring N_voxel_final.
    upsampling_type: str = "unaligned"  # choose from "aligned", "unaligned".
    nSamples: int = 1000000
    multires_sample: bool = False

    mse_loss_weight: float = 1.0

    # Scene Flow
    flow_reg_weight: float = 0.0
    flow_chain_reg_weight: float = 0.0
    static_flow_reg_weight: float = 0.0

    photo_weight: float = 0.0
    photo_chain_weight: float = 0.0
    photo_inter_weight: float = 0.0

    # kinematic regularization
    phys_kinematic_integrity: float = 0.0
    phys_higher_order_penalty: float = 0.0
    phys_transport: float = 0.0
    phys_rigidity: float = 0.0

    # hessian computation
    num_hessian_samples: int = 32
    num_grad_res_scale: float = 1.0
    grad_res_multiplier: float = 5.0

    # cycle
    flow_cycle_reg_weight: float = 0.0

    # flow loss
    flow_loss_weight: float = 0.0
    flow_loss_mask_begin: int = 0
    static_flow_loss_weight: float = 0.0
    
    # normal reg
    normal_reg_weight: float = 0.0

    # depth median loss
    depth_med_loss: bool = False
    depth_med_loss_weight: float = 0.0

    # masked losses
    static_loss_weight: float = 0.0
    dynamic_loss_weight: float = 0.0

    # entropy
    dyst_entropy_loss_weight: float = 0.0
    st_entropy_loss_weight: float = 0.0
    dyst_entropy_skewness: float = 1.0
    dyst_entropy_loss_begin: int = 0

    # ndc params (automatically overwritten)
    ndc_params: Dict = field(default_factory=lambda: {})

    # time interval params
    time_interval: float = 1.0
    max_timeinter: float = 1.0
    init_timeinter: float= 1.0
    max_timehop: float = 1.0
    init_timehop: float= 1.0
    
    # taylor mask (from low to higher order curriculum)
    use_taylor_order_mask: bool = False
    taylor_mask_unlock_list: List[int] = field(default_factory=lambda: [500, 30000, 40000, 50000, 60000])

    # use the dynamic mask from the dataset for training
    use_precomputed_dymask: bool = True

    # flow scales
    flowgrad_scale: float = 1.0
    flow_scale: float = 1.0

    ndc_system: bool = False
    
@dataclass
class Data_Config:
    datadir: str = "./data"
    dataset_name: str = "Nvidia_NDC"
    downsample: float = 1.0
    cal_fine_bbox: bool = False
    N_vis: int = -1
    time_scale: float = 1.0
    scene_bbox_min: List[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    scene_bbox_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    N_random_pose: int = 1000

    # for ndc datasets
    ndc_bd_factor: float = 0.75

    datasampler_type: str = "images"

    # For monocular dataset
    start_frame: int = -1
    end_frame: int = -1


@dataclass
class Optim_Config:
    # Learning Rate
    lr_density_grid: float = 0.02
    lr_app_grid: float = 0.02
    lr_flow_grid: float = 0.02
    lr_stdensity_grid: float = 0.002
    lr_stapp_grid: float = 0.002

    lr_density_nn: float = 0.001
    lr_app_nn: float = 0.001
    lr_stdensity_nn: float = 0.0001
    lr_stapp_nn: float = 0.0001
    lr_flow_nn: float = 0.001

    # Optimizer, Adam deault
    beta1: float = 0.9
    beta2: float = 0.99
    lr_decay_type: str = "exp"  # choose from "exp" or "cosine" or "linear"
    lr_decay_target_ratio: float = 0.1
    lr_decay_step: int = -1
    lr_upsample_reset: bool = True
    alternating_optim: bool = False
    
    # Scene Flow Training Scheme
    sf_begin: int = 0

    # Grad-based Training Sceheme
    gr_begin: int = 0

    # Dynamic Static Joint Learning Scheme
    dsj_begin: int = 0

    # physics begin
    phys_begin: int = 0

    batch_size: int = 4096
    n_iters: int = 25000
    n_static_iters: int = 25000

    flow_lr_decay_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    flow_lr_decay_rate: float = 0.5

    sigma_cont_lr_decay_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    sigma_cont_lr_decay_rate: float = 0.5

    static_loss_percentile: float = 0.5


@dataclass
class Config:
    config: Optional[str] = None
    expname: str = "default"

    render_only: bool = False
    render_train: bool = False
    render_test: bool = True
    render_path: bool = False
    render_test_poses: bool = False

    render_dolly_path: bool = False
    render_fun_path: bool = False
    render_extrapolation: bool = False

    systems: System_Config = System_Config()
    model: Model_Config = Model_Config()
    data: Data_Config = Data_Config()
    optim: Optim_Config = Optim_Config()

    use_intermediate_ckpt: bool = False
    render_nsff_dyn: bool = False
    render_nsff_dyn_short: bool = False
    debug_vf: bool = False
    debug_vf_high: bool = False
    arrow_resolution_mul: int = 2
    visualize_depth: bool = False

    light_mode: bool = False
