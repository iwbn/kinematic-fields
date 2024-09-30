from .nvidia_dynamic_scene_dataset_NDC import Nvidia_NDC_Dataset
from .nvidia_dynamic_scene_dataset_NDC_dynerf import Nvidia_NDC_Dataset_dynerf
from .nvidia_dynamic_scene_dataset_NDC_shorter import Nvidia_NDC_Dataset_Shorter


def get_train_dataset(cfg, is_stack=False):
    if cfg.data.dataset_name == "Nvidia_NDC":
        train_dataset = Nvidia_NDC_Dataset(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.ndc_bd_factor,
        )
    elif cfg.data.dataset_name == "Nvidia_NDC_dynerf":
        train_dataset = Nvidia_NDC_Dataset_dynerf(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.ndc_bd_factor,
        )
    elif cfg.data.dataset_name == "Nvidia_NDC_Shorter":
        train_dataset = Nvidia_NDC_Dataset_Shorter(
            cfg.data.datadir,
            "train",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.ndc_bd_factor,
        )
    else:
        raise NotImplementedError("No such dataset")
    return train_dataset


def get_test_dataset(cfg, is_stack=True, is_full_test=False):
    if cfg.data.dataset_name == "Nvidia_NDC":
        if is_full_test:
            test_dataset = Nvidia_NDC_Dataset(
                cfg.data.datadir,
                "full_test",
                cfg.data.downsample,
                is_stack=is_stack,
                cal_fine_bbox=cfg.data.cal_fine_bbox,
                N_vis=cfg.data.N_vis,
                time_scale=cfg.data.time_scale,
                scene_bbox_min=cfg.data.scene_bbox_min,
                scene_bbox_max=cfg.data.scene_bbox_max,
                N_random_pose=cfg.data.N_random_pose,
                bd_factor=cfg.data.ndc_bd_factor,
            )
        else:
            test_dataset = Nvidia_NDC_Dataset(
                cfg.data.datadir,
                "test",
                cfg.data.downsample,
                is_stack=is_stack,
                cal_fine_bbox=cfg.data.cal_fine_bbox,
                N_vis=cfg.data.N_vis,
                time_scale=cfg.data.time_scale,
                scene_bbox_min=cfg.data.scene_bbox_min,
                scene_bbox_max=cfg.data.scene_bbox_max,
                N_random_pose=cfg.data.N_random_pose,
                bd_factor=cfg.data.ndc_bd_factor,
            )
    elif cfg.data.dataset_name == "Nvidia_NDC_dynerf":
        test_dataset = Nvidia_NDC_Dataset_dynerf(
            cfg.data.datadir,
            "test",
            cfg.data.downsample,
            is_stack=is_stack,
            cal_fine_bbox=cfg.data.cal_fine_bbox,
            N_vis=cfg.data.N_vis,
            time_scale=cfg.data.time_scale,
            scene_bbox_min=cfg.data.scene_bbox_min,
            scene_bbox_max=cfg.data.scene_bbox_max,
            N_random_pose=cfg.data.N_random_pose,
            bd_factor=cfg.data.ndc_bd_factor,
        )
    elif cfg.data.dataset_name == "Nvidia_NDC_Shorter":
        if is_full_test:
            test_dataset = Nvidia_NDC_Dataset_Shorter(
                cfg.data.datadir,
                "full_test",
                cfg.data.downsample,
                is_stack=is_stack,
                cal_fine_bbox=cfg.data.cal_fine_bbox,
                N_vis=cfg.data.N_vis,
                time_scale=cfg.data.time_scale,
                scene_bbox_min=cfg.data.scene_bbox_min,
                scene_bbox_max=cfg.data.scene_bbox_max,
                N_random_pose=cfg.data.N_random_pose,
                bd_factor=cfg.data.ndc_bd_factor,
            )
        else:
            test_dataset = Nvidia_NDC_Dataset_Shorter(
                cfg.data.datadir,
                "test",
                cfg.data.downsample,
                is_stack=is_stack,
                cal_fine_bbox=cfg.data.cal_fine_bbox,
                N_vis=cfg.data.N_vis,
                time_scale=cfg.data.time_scale,
                scene_bbox_min=cfg.data.scene_bbox_min,
                scene_bbox_max=cfg.data.scene_bbox_max,
                N_random_pose=cfg.data.N_random_pose,
                bd_factor=cfg.data.ndc_bd_factor,
            )
    else:
        raise NotImplementedError("No such dataset")
    return test_dataset
