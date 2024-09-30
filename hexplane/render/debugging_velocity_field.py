import argparse

import imageio
import torch
import numpy as np
import cv2
from tqdm.auto import tqdm
from hexplane.model.flow_viz import flow_to_image, flow_to_image_maxmag
import os
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from hexplane.dataloader import get_test_dataset
from hexplane.model.vf_utils import OctreeRender_trilinear_scene_flow
from hexplane.render.util.util import visualize_depth_numpy
import lpips_models
import matplotlib.pyplot as plt


def save_flow_figure(save_path, img, flow, step=16):

    Y, X = np.mgrid[0:flow.shape[0], 0:flow.shape[1]]
    U = flow[...,0]
    V = flow[...,1]

    y, x = np.mgrid[step//2:flow.shape[0]:step, step//2:flow.shape[1]:step]
    u = U[y,x]
    v = V[y,x]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(img, extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.quiver(x, y, u, -v, color='r', angles='xy', scale_units='xy', scale=1.)

    # Optionally adjust the plot limits and other parameters
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())

    # Turn off the axis
    plt.axis('off')

    # Save the plot with arrows overlaid on the color image to a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0)

    #plt.show()
    # Close the plot
    plt.close()

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

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis]).permute((3, 2, 0, 1))

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
    resolution_mul=2,
    device="cuda",
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps = [], [], [], []
    strain_rates, sc_maps = [], []
    flows, vels, accs, jers = [], [], [], []
    msssims, ssims, l_alex, l_vgg = [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)


    try:
        tqdm._instances.clear()
    except Exception:
        pass

    if type(model).__name__ == "DataParallel":
            model_name = type(model.module).__name__

    if model_name.startswith("HexPlane_Flow") or model_name.startswith("HexPlaneSD"):
        renderer = OctreeRender_trilinear_scene_flow
    else:
        renderer = OctreeRender_trilinear_fast

    #test_dataset = get_test_dataset(cfg, is_stack=True, is_full_test=True)

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))

    root_dir = test_dataset.root_dir
    W, H = test_dataset.img_wh
    camera_cnt = 12

    metrics_list = ["PSNR", "SSIM", "LPIPS", "PSNR_dy", "SSIM_dy", "LPIPS_dy"]
    metrics = {k: [] for k in metrics_list}
    
    if type(model).__name__ == "DataParallel":
        model_name = type(model.module).__name__
        hexplane = model.module
    else:
        hexplane = model
    hexplane.debug_vf = True

    data = test_dataset[10]
    samples, _, sample_times = data["rays"], data["rgbs"], data["time"]
    

    all_times = np.linspace(-1, 1, 24*4)
    
    img_idx = 0
    tqdm_ = tqdm(all_times)
    for i, time in enumerate(tqdm_):
        out_path = os.path.join(f"{savePath}/{prefix}")
        os.makedirs(out_path, exist_ok=True)
        out_img_base = 'time_%03d'%(i + 1)

        rays = samples.view(-1, samples.shape[-1])
        
        times = torch.ones_like(sample_times).view(-1, sample_times.shape[-1])
        times *= time

        kwargs = {}
        if "pose" in data.keys() and renderer == OctreeRender_trilinear_scene_flow:
            kwargs['poses'] = {'pose': data["pose"]}

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
        if res_sf is None:
            res_sf = {}

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        depth_maps.append(depth_map)
        
        if savePath is not None:
            imageio.imwrite(f"{out_path}/{out_img_base}.png", rgb_map)

            if "dyrgb_map" in res_sf:
                dyrgb_map = torch.cat(res_sf['dyrgb_map']).clamp(0.0, 1.0).reshape(H,W,3).cpu()
                dyrgb_map = (dyrgb_map.numpy() * 255).astype(np.uint8)
                imageio.imwrite(f"{out_path}/{out_img_base}_dynamic.png", dyrgb_map)

            if "optical_flow_fw_ref" in res_sf.keys():
                flow_fw = torch.concat(res_sf['optical_flow_fw_ref']).reshape(H,W,2).cpu()
                flows.append(flow_fw)

                flow_img = flow_to_image(flow_fw.numpy(), arrow=True)
                imageio.imwrite(f"{out_path}/{out_img_base}_flow_fw.png", flow_img)
                #save_flow_figure(f"{out_path}/{out_img_base}_flow_fw.pdf", flow_img, flow_fw.numpy())
            
            if "rgb_warped_post" in res_sf.keys():
                rgb_warped_post = torch.concat(res_sf['rgb_warped_post']).reshape(H,W,3).cpu()
                rgb_warped_post = (rgb_warped_post.numpy() * 255).astype("uint8")
                imageio.imwrite(f"{out_path}/{out_img_base}_warped_post.png", rgb_warped_post)
            
            if "rgb_warped_prev" in res_sf.keys():
                rgb_warped_prev = torch.concat(res_sf['rgb_warped_prev']).reshape(H,W,3).cpu()
                rgb_warped_prev = (rgb_warped_prev.numpy() * 255).astype("uint8")
                imageio.imwrite(f"{out_path}/{out_img_base}_warped_prev.png", rgb_warped_prev)

            if "velocity_map_fw" in res_sf.keys():
                vel_fw = torch.concat(res_sf['velocity_map_fw']).reshape(H,W,2).cpu()
                vels.append(vel_fw)
                vel_img = flow_to_image(vel_fw.numpy(), arrow=True)
                imageio.imwrite(f"{out_path}/{out_img_base}_vel_fw.png", vel_img)
            
            if "acceleration_map_fw" in res_sf.keys():
                acc_fw = torch.concat(res_sf['acceleration_map_fw']).reshape(H,W,2).cpu()
                accs.append(acc_fw)
                acc_img = flow_to_image(acc_fw.numpy() * 10, arrow=True)
                imageio.imwrite(f"{out_path}/{out_img_base}_acc_fw.png", acc_img)

            if "jerk_map_fw" in res_sf.keys():
                jer_fw = torch.concat(res_sf['jerk_map_fw']).reshape(H,W,2).cpu()
                jers.append(jer_fw)
                jer_img = flow_to_image(jer_fw.numpy() * 100, arrow=True)
                imageio.imwrite(f"{out_path}/{out_img_base}_jer_fw.png", jer_img)

            if "strain_rate_map" in res_sf.keys():
                strain_rate_map = torch.concat(res_sf['strain_rate_map']).reshape(H,W,1).cpu()
                strain_rate_map = strain_rate_map.repeat(1,1,3)
                strain_rates.append(strain_rate_map)

                strain_rate_map = strain_rate_map / strain_rate_map.max()
                strain_rate_map = (strain_rate_map.numpy() * 255).astype("uint8")
                
                imageio.imwrite(f"{out_path}/{out_img_base}_strain.png", strain_rate_map)

            if "normal_map" in res_sf.keys():
                normal_map = torch.concat(res_sf['normal_map']).reshape(H,W,3).cpu()

                normal_map = (normal_map + 1. / 2.)
                normal_map = (normal_map.numpy() * 255).astype("uint8")
                
                imageio.imwrite(f"{out_path}/{out_img_base}_normal.png", normal_map)

            if "sigma_cont_map" in res_sf.keys():
                sigma_cont_map = torch.concat(res_sf['sigma_cont_map']).reshape(H,W,1).cpu()
                sigma_cont_map = sigma_cont_map.repeat(1,1,3)
                sc_maps.append(sigma_cont_map)

                sigma_cont_map = sigma_cont_map / strain_rate_map.max()
                sigma_cont_map = (sigma_cont_map.numpy() * 255).astype("uint8")
                
                imageio.imwrite(f"{out_path}/{out_img_base}_sc.png", sigma_cont_map)

            if "acceleration_map_fw_pred" in res_sf.keys():
                acc_fw = torch.concat(res_sf['acceleration_map_fw_pred']).reshape(H,W,2).cpu()
                acc_img = flow_to_image(acc_fw.numpy())
                imageio.imwrite(f"{out_path}/{out_img_base}_acc_pred_fw.png", acc_img)

            if "jerk_map_fw_pred" in res_sf.keys():
                jer_fw = torch.concat(res_sf['jerk_map_fw_pred']).reshape(H,W,2).cpu()
                jer_img = flow_to_image(jer_fw.numpy())
                imageio.imwrite(f"{out_path}/{out_img_base}_jer_pred_fw.png", jer_img)  


            depth_map_viz, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
            #rgb_map = np.concatenate((rgb_map, depth_map_viz), axis=1)
            imageio.imwrite(f"{out_path}/{out_img_base}_depth.png", depth_map_viz)
    
    if flows:
        flows = torch.stack(flows, dim=0)
        max_mag = torch.max(flows).numpy()
        for i, flow in enumerate(flows):
            out_img_base = 'time_%03d'%(i + 1)
            flow_img = flow_to_image_maxmag(flow.numpy(), max_mag=max_mag)
            arrow_img = flow_to_image_maxmag(flow.numpy(), max_mag=max_mag, nohsv=True, scale=resolution_mul)
            imageio.imwrite(f"{out_path}/{out_img_base}_flow_fw_norm.png", flow_img)
            #save_flow_figure(f"{out_path}/{out_img_base}_flow_fw_norm.pdf", flow_img, flow.numpy())
    if accs:
        flows = torch.stack(accs, dim=0)
        max_mag = torch.max(flows).numpy()
        for i, flow in enumerate(flows):
            out_img_base = 'time_%03d'%(i + 1)
            flow_img = flow_to_image_maxmag(flow.numpy() * 10, max_mag=max_mag * 10)
            arrow_img = flow_to_image_maxmag(flow.numpy() * 10, max_mag=max_mag * 10, nohsv=True, scale=resolution_mul)
            imageio.imwrite(f"{out_path}/{out_img_base}_acc_fw_norm.png", flow_img)
            imageio.imwrite(f"{out_path}/{out_img_base}_acc_fw_norm_arrow.png", arrow_img)
            

    if vels:
        flows = torch.stack(vels, dim=0)
        max_mag = torch.max(flows).numpy()
        for i, flow in enumerate(flows):
            out_img_base = 'time_%03d'%(i + 1)
            flow_img = flow_to_image_maxmag(flow.numpy(), max_mag=max_mag)
            arrow_img = flow_to_image_maxmag(flow.numpy(), max_mag=max_mag, nohsv=True, scale=resolution_mul)
            imageio.imwrite(f"{out_path}/{out_img_base}_vel_fw_norm.png", flow_img)
            imageio.imwrite(f"{out_path}/{out_img_base}_vel_fw_norm_arrow.png", arrow_img)

    if jers:
        flows = torch.stack(jers, dim=0)
        max_mag = torch.max(flows).numpy()
        for i, flow in enumerate(flows):
            out_img_base = 'time_%03d'%(i + 1)
            flow_img = flow_to_image_maxmag(flow.numpy() * 100, max_mag=max_mag * 100)
            arrow_img = flow_to_image_maxmag(flow.numpy() * 100, max_mag=max_mag * 100, nohsv=True, scale=resolution_mul)
            imageio.imwrite(f"{out_path}/{out_img_base}_jer_fw_norm.png", flow_img)
            imageio.imwrite(f"{out_path}/{out_img_base}_jer_fw_norm_arrow.png", arrow_img)

    if sc_maps:
        maps = torch.stack(sc_maps, dim=0)
        max_mag = torch.max(maps)
        for i, map in enumerate(maps):
            out_img_base = 'time_%03d'%(i + 1)
            map = map / max_mag
            map = (map.numpy() * 255).astype("uint8")
            
            imageio.imwrite(f"{out_path}/{out_img_base}_sc_norm.png", map)

    if strain_rates:
        maps = torch.stack(strain_rates, dim=0)
        max_mag = torch.max(maps)
        for i, map in enumerate(maps):
            out_img_base = 'time_%03d'%(i + 1)
            map = map / max_mag
            map = (map.numpy() * 255).astype("uint8")
            
            imageio.imwrite(f"{out_path}/{out_img_base}_strain_norm.png", map)

    return metrics['PSNR']
            


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True, channel_axis=-1, data_range=1.0)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid