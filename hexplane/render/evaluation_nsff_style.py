import argparse

import imageio
import torch
import numpy as np
import cv2
from tqdm.auto import tqdm
from hexplane.model.flow_viz import flow_to_image
import os
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from hexplane.dataloader import get_test_dataset
from hexplane.model.vf_utils import OctreeRender_trilinear_scene_flow
from hexplane.render.util.util import visualize_depth_numpy
import lpips_models


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

    if model_name.startswith("HexPlane_Flow") or model_name.startswith("HexPlaneSD"):
        renderer = OctreeRender_trilinear_scene_flow
    else:
        renderer = OctreeRender_trilinear_fast

    test_dataset = get_test_dataset(cfg, is_stack=True, is_full_test=True)

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
    idxs = list(range(0, len(test_dataset), img_eval_interval))
    count = int(np.round(2.0*test_dataset.time_scale/test_dataset.frame_interval + 1.))

    root_dir = test_dataset.root_dir
    W, H = test_dataset.img_wh
    camera_cnt = 12

    metrics_list = ["PSNR", "SSIM", "LPIPS", "PSNR_dy", "SSIM_dy", "LPIPS_dy"]
    metrics = {k: [] for k in metrics_list}

    lpips_model = lpips_models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)
    
    iter_list = []
    for camera_i in tqdm(range(camera_cnt)):
        for img_i in range(count):
            if img_i % 12 == camera_i:
                continue
            iter_list.append((camera_i, img_i))

    assert len(test_dataset) == len(iter_list)
    

    if "start_frame" not in test_dataset.__dict__ or test_dataset.start_frame < 0:
        start_frame = 0
    else:
        start_frame = test_dataset.start_frame

    img_idx = 0
    tqdm_ = tqdm(iter_list)
    for camera_i, img_i in tqdm_:
        img_i += start_frame
        # gt_img_path = os.path.join(root_dir, 
        #                     'mv_images', 
        #                     '%05d'%img_i, 
        #                     'cam%02d.jpg'%(camera_i + 1))

        dyn_mask_path = os.path.join(root_dir, 
                                    'mv_masks', 
                                    '%05d'%img_i, 
                                    'cam%02d.png'%(camera_i + 1))
        
        out_path = os.path.join(f"{savePath}/{prefix}", '%05d'%img_i)
        out_img_base = 'cam%02d'%(camera_i + 1)
        os.makedirs(out_path, exist_ok=True)
        
        # gt_img = cv2.imread(gt_img_path)[:, :, ::-1] / 255.
        # gt_img = cv2.resize(gt_img, 
        #                     dsize=(W, H), 
        #                     interpolation=cv2.INTER_AREA)

        dy_mask_gt = np.float32(cv2.imread(dyn_mask_path) > 1e-3)#/255.
        dy_mask_gt = cv2.resize(dy_mask_gt[...,0:1], 
                            dsize=(W, H), 
                            interpolation=cv2.INTER_NEAREST)
        
        data = test_dataset[img_idx]
        samples, gt_rgb, sample_times = data["rays"], data["rgbs"], data["time"]
        #img_file_path = test_dataset.all_paths[img_idx]

        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])

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
        gt_rgb = gt_rgb.view(H, W, 3)
        #mask_dy = torch.cat(res_sf['mask_dy']).clamp(0.0, 1.0).reshape(H,W,1).cpu()

        gt_rgb = gt_rgb.clamp(0.0, 1.0)

        psnr = peak_signal_noise_ratio(gt_rgb.numpy(), rgb_map.numpy()) # skimage.measure.compare_psnr(gt_img, rgb)
        ssim = structural_similarity(gt_rgb.numpy(), rgb_map.numpy(), multichannel=True, channel_axis=-1, data_range=1.0) # skimage.measure.compare_ssim(gt_img, rgb, 
                                            #multichannel=True)
        
        gt_img_0 = im2tensor(gt_rgb).cuda()
        rgb_0 = im2tensor(rgb_map).cuda()

        lpips = lpips_model.forward(gt_img_0, rgb_0)
        lpips = lpips.item()
        tqdm_.set_postfix(file=str(('%05d'%img_i, out_img_base)), psnr=psnr, ssim=ssim, lpips=lpips)

        metrics['PSNR'].append(psnr)
        metrics['SSIM'].append(ssim)
        metrics['LPIPS'].append(lpips)


        dynamic_mask_0 = torch.Tensor(dy_mask_gt[:, :, np.newaxis, np.newaxis].transpose((3, 2, 0, 1)))

        dynamic_ssim = calculate_ssim(gt_rgb.numpy(), rgb_map.numpy(), dy_mask_gt[...,None])
        dynamic_psnr = calculate_psnr(gt_rgb.numpy(), rgb_map.numpy(), dy_mask_gt[...,None])

        dynamic_lpips = lpips_model.forward(gt_img_0, 
                                        rgb_0, 
                                        dynamic_mask_0).item()

        metrics['PSNR_dy'].append(dynamic_psnr)
        metrics['SSIM_dy'].append(dynamic_ssim)
        metrics['LPIPS_dy'].append(dynamic_lpips)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        #mask_dy = (mask_dy.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{out_path}/{out_img_base}.png", rgb_map)
            #imageio.imwrite(f"{out_path}/{out_img_base}_dy.png", mask_dy)
            imageio.imwrite(f"{out_path}/{out_img_base}_gt.png", gt_rgb_map)

            if "optical_flow_fw_ref" in res_sf.keys():
                flow_fw = torch.concat(res_sf['optical_flow_fw_ref']).reshape(H,W,2).cpu()
                flow_img = flow_to_image(flow_fw.numpy())
                imageio.imwrite(f"{out_path}/{out_img_base}_flow_fw.png", flow_img)
            depth_map_viz, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
            #rgb_map = np.concatenate((rgb_map, depth_map_viz), axis=1)
            imageio.imwrite(f"{out_path}/{out_img_base}_depth.png", depth_map_viz)

        img_idx += 1
        if img_idx > len(test_dataset):
            break

    with open(f"{savePath}/{prefix}mean.txt", "w") as f:
        res = ""
        for k in metrics_list:
            res += f"{k}: {np.mean(np.asarray(metrics[k]))}, "
        res = res[:-2] + "\n"

        f.write(
            res
        )
        print(
            res
        )
        for i in range(img_idx):
            res = "Index %d, " % i
            for k in metrics_list:
                res += f"{k}: {metrics[k][i]}, "
            res = res[:-2] + "\n"
            f.write(
                res
            )

    return metrics['PSNR']
            


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    mask = np.ones_like(img1) * mask  # broadcast shape

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
    mask = np.ones_like(img1) * mask  # broadcast shape
    
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True, channel_axis=-1, data_range=1.0)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid