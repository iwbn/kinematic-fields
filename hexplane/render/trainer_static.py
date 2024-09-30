import os
import math
import sys
import glob
from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm

from hexplane.render.render import evaluation
from hexplane.render.util.Sampling import cal_n_samples
from hexplane.render.util.util import N_to_reso
from .trainer import Trainer, LOG_DETAILS

from hexplane.model.vf_utils import OctreeRender_trilinear_scene_flow

# frequently used function
_c = lambda x: torch.cat(x, dim=0)

class StaticTrainer(Trainer):
    def sample_data(self, train_dataset, iteration):
        """
        Sample a batch of data from the dataset, but only static pixels.
        """
        train_depth = None
        train_pose = None
        train_flow = None
        train_dymask = None
        
        with torch.no_grad():
            if True:
                if "st_samples" in train_dataset.__dict__:
                    all_st_rays = train_dataset.all_st_rays
                    all_st_rgbs = train_dataset.all_st_rgbs
                    all_st_depths = train_dataset.all_st_depths

                    all_st_poses = train_dataset.all_st_poses
                    all_st_poses_post = train_dataset.all_st_poses_post
                    all_st_poses_prev = train_dataset.all_st_poses_prev
                    all_st_flows = train_dataset.all_st_flows
                else:
                    st_samples = (train_dataset.all_dymasks[:,0] < 1e-3)
                    num_st_samples = st_samples.sum().float()

                    all_st_rays = train_dataset.all_rays[st_samples].view(-1, 6)
                    all_st_rgbs = train_dataset.all_rgbs[:,0][st_samples].view(-1, 3)
                    all_st_depths = train_dataset.all_depths[:,0][st_samples].view(-1)
                    all_st_flows = train_dataset.all_flows[:,0][st_samples].view(-1, 2, 3)


                    st_poses = train_dataset.all_poses
                    st_poses = st_poses.broadcast_to([st_poses.shape[0], st_samples.shape[1], 
                                                      st_poses.shape[-2], st_poses.shape[-1]])

                    st_poses_post = torch.roll(st_poses, -1, 0)
                    st_poses_prev = torch.roll(st_poses, 1, 0)

                    all_st_poses = st_poses[st_samples]
                    all_st_poses_post = st_poses_post[st_samples]
                    all_st_poses_prev = st_poses_prev[st_samples]


                    train_dataset.all_st_rays = torch.FloatTensor(all_st_rays)
                    train_dataset.all_st_rgbs = torch.FloatTensor(all_st_rgbs)
                    train_dataset.all_st_depths = torch.FloatTensor(all_st_depths)

                    train_dataset.all_st_poses = torch.FloatTensor(all_st_poses)
                    train_dataset.all_st_poses_post = torch.FloatTensor(all_st_poses_post)
                    train_dataset.all_st_poses_prev = torch.FloatTensor(all_st_poses_prev)
                    train_dataset.all_st_flows = torch.FloatTensor(all_st_flows)

                    train_dataset.st_samples = st_samples
                    train_dataset.num_st_samples = num_st_samples

                if 'all_dymasks' not in train_dataset.__dict__:
                    raise ValueError ("all_dymask should be enabled in the dataset")

                rays_train = all_st_rays
                select_inds = torch.randint(low=0, high=rays_train.shape[0], 
                                            size=[self.cfg.optim.batch_size],
                                            device=rays_train.device)

                rays_train = rays_train[select_inds].to(self.device)
                rgb_train = all_st_rgbs[select_inds].to(self.device)
                train_depth = all_st_depths[select_inds].to(self.device)

                train_pose = {'pose': all_st_poses[select_inds].to(self.device),
                              'post_pose': all_st_poses_post[select_inds].to(self.device),
                              'prev_pose': all_st_poses_prev[select_inds].to(self.device),
                              }

                train_flow = all_st_flows[select_inds].to(self.device)

                frame_time = torch.zeros_like(rgb_train[...,:1])

            if not self.cfg.model.use_dymask_gt:
                raise ValueError("enable use_dymask_gt")
        
        return rays_train, rgb_train, frame_time, train_depth, train_pose, train_flow, train_dymask

    def train(self):
        # load the training and testing dataset and other settings.
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        model = self.model
        summary_writer = self.summary_writer
        reso_cur = self.reso_cur

        if type(model).__name__ == "DataParallel":
            model_name = type(model.module).__name__
            hexplane = model.module
        else:
            hexplane = model

        renderer = OctreeRender_trilinear_scene_flow # renderer with flow field

        ndc_ray = train_dataset.ndc_ray  # if the rays are in NDC
        white_bg = test_dataset.white_bg  # if the background is white

        # Calculate the number of samples for each ray based on the current resolution.
        nSamples = min(
            self.cfg.model.nSamples,
            cal_n_samples(reso_cur, self.cfg.model.step_ratio),
        )

        # initialize the data sampler
        self.init_sampler(train_dataset)
        # precompute the voxel upsample list
        self.get_voxel_upsample_list()

        if type(model).__name__ == "DataParallel":
            model_name = type(model.module).__name__
            hexplane = model.module
        else:
            hexplane = model

        upsample_list = deepcopy(self.cfg.model.upsample_list)
        N_voxel_list = deepcopy(self.N_voxel_list)
        Time_grid_list = deepcopy(self.Time_grid_list)

        # Initialiaze TV loss on planse
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        tvreg_s_t = TVLoss(
            1.0, self.cfg.model.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes

        # Initialize the optimizers
        grad_vars = hexplane.get_optparam_groups(self.cfg.optim)
        optimizer = torch.optim.Adam(
            grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
        )

        reso_cur = self.reso_cur

        # Calculate the number of samples for each ray based on the current resolution.
        nSamples = min(
            self.cfg.model.nSamples,
            cal_n_samples(reso_cur, self.cfg.model.step_ratio),
        )

        pbar = tqdm(
            range(self.cfg.optim.n_static_iters),
            miniters=self.cfg.systems.progress_refresh_rate,
            file=sys.stdout,
        )

        if self.cfg.optim.n_static_iters <= 0:
            return 
        iteration_ratio = self.cfg.optim.n_iters / self.cfg.optim.n_static_iters
        upsample_list = [i // iteration_ratio for i in upsample_list]

        PSNRs = []
        PSNRs_test = []
        for iteration in pbar:
            # flow lr decay (applied to flow and depth losses)
            decay_count = (np.array(self.cfg.optim.flow_lr_decay_list) // iteration_ratio < iteration).sum()
            flow_lr_decay = 1.0 * (self.cfg.optim.flow_lr_decay_rate ** decay_count)

            # Sample dat
            rays_train, rgb_train, frame_time, depth, pose, flow, dymask_gt = self.sample_data(
                train_dataset, iteration
            )

            kwargs = {}
            if pose is not None:
                if isinstance(pose, dict):
                    kwargs['poses'] = pose
                else:
                    raise NotImplementedError ("please check the flow inputs; that should be a dict")

            # Render the rgb values of rays
            rgb_map, alphas_map, depth_map, weights, aux = renderer(
                rays_train,
                frame_time,
                model,
                chunk=self.cfg.optim.batch_size,
                N_samples=nSamples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                device=self.device,
                is_train=True,
                **kwargs
            )

            # Calculate the loss
            loss = torch.mean((rgb_map - rgb_train) ** 2)
            total_loss = loss

            # Loss on the rendered and gt depth maps.
            if self.cfg.model.depth_loss and self.cfg.model.depth_loss_weight > 0:
                depth_loss = (depth_map - depth) ** 2
                mask = depth != 0
                depth_loss = torch.mean(depth_loss[mask])
                total_loss += depth_loss * self.cfg.model.depth_loss_weight
                summary_writer.add_scalar(
                    "train/st_depth_loss",
                    depth_loss.detach().item(),
                    global_step=iteration,
                )

            if flow is not None and self.cfg.model.static_flow_loss_weight > 0:
                flow_fw_gt = flow[..., 0, :2]
                flow_fw_gt_mask = flow[..., 0, 2:]
                flow_bw_gt = flow[..., 1, :2]
                flow_bw_gt_mask = flow[..., 1, 2:]
                
                loss_post = torch.mean(flow_fw_gt_mask * torch.abs(flow_fw_gt - _c(aux['optical_flow_fw_post'])))
                loss_prev = torch.mean(flow_bw_gt_mask * torch.abs(flow_bw_gt - _c(aux['optical_flow_bw_prev'])))

                flow_loss = (loss_post + loss_prev)
                total_loss += flow_loss * self.cfg.model.static_flow_loss_weight * flow_lr_decay
                summary_writer.add_scalar(
                    "train/st_flow_loss", flow_loss.detach().item(), global_step=iteration
                )

            # Calculate the learning rate decay factor
            lr_factor = self.get_lr_decay_factor(int(iteration * iteration_ratio))

            # regularization
            # TV loss on the density planes
            if self.cfg.model.TV_weight_stdensity > 0:
                TV_weight_density = lr_factor * self.cfg.model.TV_weight_stdensity
                loss_tv = hexplane.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_stdensity",
                    loss_tv.detach().item(),
                    global_step=iteration,
                )

            # TV loss on the appearance planes
            if self.cfg.model.TV_weight_stapp > 0:
                TV_weight_app = lr_factor * self.cfg.model.TV_weight_stapp
                loss_tv = hexplane.TV_loss_app(tvreg_s, tvreg_s_t) * TV_weight_app
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_stapp", loss_tv.detach().item(), global_step=iteration
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar("train/st_PSNR", PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar("train/st_mse", loss, global_step=iteration)

            # Print the current values of the losses.
            if iteration % self.cfg.systems.progress_refresh_rate == 0:
                pbar.set_description(
                    f"Iteration {iteration:05d}:"
                    + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                    + f" mse = {loss:.6f}"
                )
                PSNRs = []

            # Decay the learning rate.
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr_org"] * lr_factor

            if (
                iteration % self.cfg.systems.vis_every == self.cfg.systems.vis_every - 1
                and self.cfg.data.N_vis != 0
            ):
                PSNRs_test = evaluation(
                    test_dataset,
                    model,
                    self.cfg,
                    f"{self.logfolder}/imgs_vis_st/",
                    self.cfg.data.N_vis,
                    prefix=f"{iteration:06d}_",
                    white_bg=white_bg,
                    N_samples=nSamples,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    compute_extra_metrics=False,
                )
                summary_writer.add_scalar(
                    "test/st_psnr", np.mean(PSNRs_test), global_step=iteration
                )

                torch.cuda.synchronize()

            # Upsample the volume grid.
            if iteration in upsample_list:
                if self.cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                else:
                    N_voxel = N_voxel_list.pop(0)
                    reso_cur = N_to_reso(
                        N_voxel, hexplane.aabb, self.cfg.model.nonsquare_voxel
                    )
                time_grid = Time_grid_list.pop(0)
                nSamples = min(
                    self.cfg.model.nSamples,
                    cal_n_samples(reso_cur, self.cfg.model.step_ratio),
                )
                hexplane.upsample_volume_grid(reso_cur, time_grid)

                grad_vars = hexplane.get_optparam_groups(self.cfg.optim, 1.0)
                optimizer = torch.optim.Adam(
                    grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
                )