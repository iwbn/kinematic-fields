import os

import imageio
import numpy as np
import torch
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm
import cv2

from hexplane.render.util.metric import rgb_lpips, rgb_ssim
from hexplane.render.util.util import visualize_depth_numpy
from hexplane.model.vf_utils import OctreeRender_trilinear_scene_flow
from hexplane.dataloader.ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender

from hexplane.model.flow_viz import flow_to_image


def OctreeRender_trilinear_fast(
    rays,
    time,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    device="cuda",
    **kwargs
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals = [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map, alpha_map, z_val_map = model(
            rays_chunk,
            time_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
        )
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        None,
    )


@torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps = [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    if type(model).__name__ == "DataParallel":
        model_name = type(model.module).__name__
        hexplane = model.module
    else:
        model_name = type(model).__name__
        hexplane = model

    hexplane.render_kinematic_field = True

    if model_name.startswith("HexPlane_Flow") or model_name.startswith("HexPlaneSD") or model_name.startswith("TensoRF_Slim"):
        renderer = OctreeRender_trilinear_scene_flow
    else:
        renderer = OctreeRender_trilinear_fast

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    for idx in tqdm(idxs):
        data = test_dataset[idx]
        samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]

        depth = None

        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])

        kwargs = {}
        if "pose" in data.keys():
            kwargs['poses'] = {'pose': data["pose"]}
        
        if "post_pose" in data.keys():
            kwargs['poses']['post_pose'] = data['post_pose']

        if "prev_pose" in data.keys():
            kwargs['poses']['prev_pose'] = data['prev_pose']

        rgb_map, _, depth_map, _, res_sf = renderer(
            rays,
            times,
            model,
            chunk=4096,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            **kwargs
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        if res_sf is None:
            res_sf = {}

        
        if "dy_mask" in res_sf:
            mask_dy = torch.cat(res_sf['dy_mask']).clamp(0.0, 1.0).reshape(H,W,1).cpu()
        else:
            mask_dy = torch.ones_like(rgb_map[...,0])

        if "dy_alpha_mask" in res_sf:
            dy_alpha_mask = torch.cat(res_sf['dy_alpha_mask']).clamp(0.0, 1.0).reshape(H,W,1).cpu()
            dy_alpha_mask = (dy_alpha_mask.numpy() * 255).astype(np.uint8)

        if "st_alpha_mask" in res_sf:
            st_alpha_mask = torch.cat(res_sf['st_alpha_mask']).clamp(0.0, 1.0).reshape(H,W,1).cpu()
            st_alpha_mask = (st_alpha_mask.numpy() * 255).astype(np.uint8)

        if "dyrgb_map" in res_sf:
            dyrgb_map = torch.cat(res_sf['dyrgb_map']).clamp(0.0, 1.0).reshape(H,W,3).cpu()
            dyrgb_map = (dyrgb_map.numpy() * 255).astype(np.uint8)

        if "strgb_map" in res_sf:
            strgb_map = torch.cat(res_sf['strgb_map']).clamp(0.0, 1.0).reshape(H,W,3).cpu()
            strgb_map = (strgb_map.numpy() * 255).astype(np.uint8)

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depth" in data.keys():
            depth = data["depth"]
            gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if "stdepth_map" in res_sf:
            stdepth_map = torch.cat(res_sf['stdepth_map']).reshape(H,W).cpu()
            stdepth_map, _ = visualize_depth_numpy(stdepth_map.numpy(), near_far)

        if "dydepth_map" in res_sf:
            dydepth_map = torch.cat(res_sf['dydepth_map']).reshape(H,W).cpu()
            dydepth_map, _ = visualize_depth_numpy(dydepth_map.numpy(), near_far)


        if len(test_dataset):
            gt_rgb = gt_rgb.view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)
                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype(np.uint8)
        mask_dy = (mask_dy.numpy() * 255).astype(np.uint8)
        gt_rgb_map = (gt_rgb.numpy() * 255).astype(np.uint8)

        if depth is not None:
            gt_depth_maps.append(gt_depth)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            _rgb = lambda x: np.tile(x, [1,1,3])[...,:3]
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_gt.png", gt_rgb_map)

            if "optical_flow_fw_ref" in res_sf:
                flow_fw = torch.concat(res_sf['optical_flow_fw_ref']).reshape(H,W,2).cpu()
                flow_img = flow_to_image(flow_fw.numpy(), arrow=True)
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_flow_fw.png", flow_img)
                #np.save(f"{savePath}/{prefix}{idx:03d}_flow_fw.npy", flow_fw.numpy())
            
            if "optical_flow_fw_post" in res_sf:
                flow_fw = torch.concat(res_sf['optical_flow_fw_post']).reshape(H,W,2).cpu()
                flow_img = flow_to_image(flow_fw.numpy(), arrow=True)
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_flow_fw_post.png", flow_img)

            if "flows" in data.keys():
                flow_fw = data['flows'][...,0,:2].reshape(H,W,2).cpu()
                flow_img = flow_to_image(flow_fw.numpy(), arrow=True)
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_flow_fw_gt.png", flow_img)
            

            if cfg.light_mode is False:
                if "dy_mask" in res_sf:
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_dy.png", _rgb(mask_dy))
                
                if "dy_alpha_mask" in res_sf:
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_dyalpha.png", _rgb(dy_alpha_mask))

                if "st_alpha_mask" in res_sf:
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_stalpha.png", _rgb(st_alpha_mask))

                if "flows" in data.keys():
                    flow_bw = data['flows'][...,1,:2].reshape(H,W,2).cpu()
                    flow_img = flow_to_image(flow_bw.numpy(), arrow=True)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_flow_bw_gt.png", flow_img)

                if "optical_flow_bw_prev" in res_sf:
                    flow_bw = torch.concat(res_sf['optical_flow_bw_prev']).reshape(H,W,2).cpu()
                    flow_img = flow_to_image(flow_bw.numpy(), arrow=True)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_flow_bw_prev.png", flow_img)

                if "velocity_map_fw" in res_sf.keys():
                    vel_fw = torch.concat(res_sf['velocity_map_fw']).reshape(H,W,2).cpu()
                    vel_img = flow_to_image(vel_fw.numpy(), arrow=True)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_vel_fw.png", vel_img)
                    #np.save(f"{savePath}/{prefix}{idx:03d}_vel_fw.npy", vel_fw.numpy())
                    

                if "acceleration_map_fw" in res_sf.keys():
                    acc_fw = acc_fw_net = torch.concat(res_sf['acceleration_map_fw']).reshape(H,W,2).cpu()
                    acc_fw_pred = torch.concat(res_sf['acceleration_map_fw_pred']).reshape(H,W,2).cpu()
                    acc_fw = torch.concat([acc_fw, acc_fw_pred], axis=1)
                    acc_img = flow_to_image(acc_fw.numpy(), arrow=True)

                    acc_fw_pred /= acc_fw_pred.abs().max() / 30.0
                    acc_fw_net /= acc_fw_net.abs().max() / 30.0
                    acc_pred_img = flow_to_image(acc_fw_pred.numpy(), arrow=True, scale=1)
                    acc_net_img = flow_to_image(acc_fw_net.numpy(), arrow=True, scale=1)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_acc_fw.png", acc_img)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_acc_fw_pred.png", acc_pred_img)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_acc_fw_net.png", acc_net_img)
                    #np.save(f"{savePath}/{prefix}{idx:03d}_acc_fw.npy", acc_fw.numpy())
                
                if "jerk_map_fw" in res_sf.keys():
                    jer_fw = jer_fw_net = torch.concat(res_sf['jerk_map_fw']).reshape(H,W,2).cpu()
                    jer_fw_pred = torch.concat(res_sf['jerk_map_fw_pred']).reshape(H,W,2).cpu()

                    jer_fw_pred /= jer_fw_pred.abs().max() / 30.0
                    jer_fw_net /= jer_fw_net.abs().max() / 30.0

                    jer_pred_img = flow_to_image(acc_fw_pred.numpy(), arrow=True, scale=1)
                    jer_net_img = flow_to_image(acc_fw_net.numpy(), arrow=True, scale=1)

                    jer_fw = torch.concat([jer_fw, jer_fw_pred], axis=1)
                    jer_img = flow_to_image(jer_fw.numpy(), arrow=True)

                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_jer_fw.png", jer_img)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_jer_fw_pred.png", jer_pred_img)
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_jer_fw_net.png", jer_net_img)
                    #np.save(f"{savePath}/{prefix}{idx:03d}_jer_fw_net.npy", jer_fw_net.numpy())
                    #np.save(f"{savePath}/{prefix}{idx:03d}_jer_fw.npy", jer_fw_pred.numpy())
                    

                if "strain_rate_map" in res_sf.keys():
                    strain_rate_map = torch.concat(res_sf['strain_rate_map']).reshape(H,W,1).cpu()
                    strain_rate_map = strain_rate_map.repeat(1,1,3)
                    strain_rate_map = strain_rate_map / strain_rate_map.max()
                    strain_rate_map = (strain_rate_map.numpy() * 255).astype("uint8")
                    
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_strain.png", strain_rate_map)

                if "normal_map" in res_sf.keys():
                    normal_map = torch.concat(res_sf['normal_map']).reshape(H,W,3).cpu()

                    normal_map = (normal_map + 1. / 2.)
                    normal_map = (normal_map.numpy() * 255).astype("uint8")
                    
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_normal.png", normal_map)

                if "pred_normal_map" in res_sf.keys():
                    pred_normal_map = torch.concat(res_sf['pred_normal_map']).reshape(H,W,3).cpu()

                    pred_normal_map = (pred_normal_map + 1. / 2.)
                    pred_normal_map = (pred_normal_map.numpy() * 255).astype("uint8")
                    
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_prednormal.png", pred_normal_map)

                if "sigma_cont_map" in res_sf.keys():
                    sigma_cont_map = torch.concat(res_sf['sigma_cont_map']).reshape(H,W,1).cpu()
                    sigma_cont_map = sigma_cont_map.repeat(1,1,3)
                    sigma_cont_map = sigma_cont_map / sigma_cont_map.max()
                    sigma_cont_map = (sigma_cont_map.numpy() * 255).astype("uint8")
                    
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_sc.png", sigma_cont_map)

            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
            if depth is not None:
                rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}_gt.png", rgb_map)

            if "dyrgb_map" in res_sf:
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_dynamic.png", dyrgb_map)

            if "strgb_map" in res_sf:
                imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_static.png", strgb_map)
            
            
            if cfg.light_mode is False:
                if "stdepth_map" in res_sf:
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_stdepth.png", stdepth_map)

                if "dydepth_map" in res_sf:
                    imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_dydepth.png", dydepth_map)
    
    hexplane.render_kinematic_field = False

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=30,
        quality=8,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps),
        fps=30,
        quality=8,
    )
    if depth is not None:
        imageio.mimwrite(
            f"{savePath}/{prefix}_gt_depthvideo.mp4",
            np.stack(gt_depth_maps),
            fps=30,
            quality=8,
        )

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}\n"
                )
                for i in range(len(PSNRs)):
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}\n"
                    )
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs


@torch.no_grad()
def evaluation_path(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    if type(model).__name__ == "DataParallel":
        model_name = type(model.module).__name__

    if model_name.startswith("HexPlane_Flow") or model_name.startswith("HexPlaneSD"):
        renderer = OctreeRender_trilinear_scene_flow
    else:
        renderer = OctreeRender_trilinear_fast

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times = test_dataset.get_val_rays()

    for idx in tqdm(range(val_times.shape[0])):
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        time = time.expand(rays.shape[0], 1)
        
        rgb_map, _, depth_map, _, _ = renderer(
            rays,
            time,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=30, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=30, quality=8
    )

    return 0

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses
