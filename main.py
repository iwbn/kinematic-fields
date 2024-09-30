import datetime
import os
import random

import numpy as np
import torch

# for faster training and inference
torch.set_float32_matmul_precision('high')


from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from hexplane.dataloader import get_test_dataset, get_train_dataset
from hexplane.model import init_model
from hexplane.render.render import evaluation, evaluation_path
from hexplane.render.trainer import Trainer
from hexplane.render.util.util import N_to_reso
from hexplane.render.util.Sampling import GM_Resi, cal_n_samples
from hexplane.model.model_helper import update_model_param

# evaluation and debugging
from hexplane.render.evaluation_nsff_style import evaluation as evaluation_nsff
from hexplane.render.evaluation_nsff_style_short import evaluation as evaluation_nsff_short
#from visualize.extract_point import evaluation as extract_point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# rendering only
def render_test(cfg):
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg

    # update model params based on the dataset
    update_model_param(cfg, test_dataset)

    # intermediate_ckpt means checkpoints created during training (not a final model)
    # init model.
    aabb = test_dataset.scene_bbox.numpy()
    near_far = test_dataset.near_far
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)

    # init trainer.
    trainer = Trainer(
        HexPlane,
        cfg,
        reso_cur,
        test_dataset,
        test_dataset,
        None,
        None,
        device,
    )

    # support data parallel
    if type(HexPlane).__name__ == "DataParallel":
        model_name = type(HexPlane.module).__name__
        hexplane = HexPlane.module
    else:
        hexplane = HexPlane

    print(hexplane)
    
    trainer.get_voxel_upsample_list()
    nSamples = min(
        cfg.model.nSamples,
        cal_n_samples(trainer.reso_cur, cfg.model.step_ratio),
    )

    loaded = trainer.load_hexplane(ckpt_path=cfg.systems.ckpt, 
                                    hexplane=hexplane, 
                                    optimizer=None, 
                                    flow_optimizer=None, 
                                    reso_cur=trainer.reso_cur, 
                                    nSamples=nSamples)

    hexplane, _, _, reso_cur, nSamples, start = loaded

    logfolder = os.path.dirname(cfg.systems.ckpt)
    logfolder = os.path.join(logfolder, "step_%d" % (start - 1))

    # render training view
    if cfg.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # render test view
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    
    # NVIDIA test (NSFF protocol)
    if cfg.data.dataset_name.startswith("Nvidia_NDC") and \
        cfg.render_nsff_dyn and cfg.data.dataset_name != "Nvidia_NDC_Shorter" and \
        cfg.data.dataset_name != "Nvidia_NDC_dynerf":
        os.makedirs(f"{logfolder}/imgs_test_nsff", exist_ok=True)
        evaluation_nsff(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_nsff/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
    
    # NVIDIA test (NSFF protocol in a short sequence (12 frame training))
    if cfg.data.dataset_name.startswith("Nvidia_NDC") and \
        (cfg.render_nsff_dyn_short or (cfg.data.dataset_name == "Nvidia_NDC_Shorter" and cfg.render_nsff_dyn)) and \
        cfg.data.dataset_name != "Nvidia_NDC_dynerf":
        print("Test on shorter sequence")
        os.makedirs(f"{logfolder}/imgs_test_nsff_shorter", exist_ok=True)
        evaluation_nsff_short(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_nsff_shorter/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # renderining with predefined path
    if cfg.render_path:
        os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

# train on training set
def reconstruction(cfg):
    
    # set OmegaConf modifiable
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)

    # load training dataset
    if cfg.data.datasampler_type == "rays":
        train_dataset = get_train_dataset(cfg, is_stack=False)
    else:
        train_dataset = get_train_dataset(cfg, is_stack=True)

    # get test dataset
    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    near_far = test_dataset.near_far
    
    # add timestamp
    if cfg.systems.add_timestamp:
        logfolder = f'{cfg.systems.basedir}/{cfg.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}"

    # update model params based on the dataset
    update_model_param(cfg, train_dataset)

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    
    # init model.
    aabb = train_dataset.scene_bbox.numpy()#.to(device)
    HexPlane, reso_cur = init_model(cfg, aabb, near_far, device)

    # init trainer
    trainer = Trainer(
        HexPlane,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    )

    # train model on training set
    trainer.train()

    # get inner hexplane to support DataParallel
    if type(HexPlane).__name__ == "DataParallel":
        model_name = type(HexPlane.module).__name__
        hexplane = HexPlane.module
    else:
        hexplane = HexPlane

    # save the final output
    torch.save(hexplane, f"{logfolder}/{cfg.expname}.th")

    # Render training viewpoints.
    if cfg.render_train:
        train_dataset = get_train_dataset(cfg, is_stack=True)
        evaluation(
            train_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_train_all/",
            prefix="train",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render test viewpoints.
    if cfg.render_test:
        evaluation(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

    # Render validation viewpoints.
    if cfg.render_path:
        evaluation_path(
            test_dataset,
            HexPlane,
            cfg,
            f"{logfolder}/imgs_path_all/",
            prefix="validation",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )


if __name__ == "__main__":
    # Load config file from base config, yaml and cli.
    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only:
        # Inference only.
        render_test(cfg)
    else:
        # Reconstruction and Inference.
        reconstruction(cfg)
