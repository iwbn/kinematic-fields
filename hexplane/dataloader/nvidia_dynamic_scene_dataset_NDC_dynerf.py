import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def read_MiDaS_disp(disp_fi, disp_rescale=10., h=None, w=None):
    disp = np.load(disp_fi)
    return disp

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


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


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


def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    while video_frames.isOpened():
        ret, video_frame = video_frames.read()
        if ret:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_frame = Image.fromarray(video_frame)
            if downsample != 1.0:
                img = video_frame.resize(img_wh, Image.LANCZOS)
            img = transform(img)
            video_data_save[count] = img.view(3, -1).permute(1, 0)
            count += 1
        else:
            break
    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] * img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs


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


class Nvidia_NDC_Dataset_dynerf(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=2.0,
        eval_step=1,
        eval_index=0,
        sphere_scale=1.0,
    ):
        self.img_wh = (
            int(540 / downsample),
            int(288 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.split = split
        self.downsample = 1.0
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = True
        self.flow_data = True

        self.load_meta()
        print("meta data loaded")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]

        near_original = self.near_fars.min()
        # Rescale if bd_factor is provided
        sc = 1. if self.bd_factor is None else 1./(near_original * self.bd_factor)
        poses[:,:3,3] *= sc
        self.near_fars *= sc

        images = glob.glob(os.path.join(self.image_dir, "*.png"))
        images.sort() # sort the list; important
        assert len(images) == poses_arr.shape[0]
        image0 = cv2.imread(images[0])
        self.img_wh = (image0.shape[1], image0.shape[0])

        fwd_paths = sorted(glob.glob(os.path.join(self.root_dir, 'flow', '*_fwd.npz')))
        bwd_paths = sorted(glob.glob(os.path.join(self.root_dir, 'flow', '*_bwd.npz')))

        disp_paths = sorted(glob.glob(os.path.join(self.root_dir, 'disp', '*.npy')))
        dymask_paths = sorted(glob.glob(os.path.join(self.root_dir, 'motion_masks', '*.png'))) 

        H, W, focal = poses[0, :, -1]
        factor = W / self.img_wh[0]

        # override original
        W, H = self.img_wh
        focal = focal / factor

        # assumes square pixels
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        poses = np.float32(poses)
        poses, pose_avg = center_poses(
            poses, self.blender2opencv
        )  # Re-center poses so that the average is near the center.

        self.poses_raw = poses

        # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 120
        self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)

        self.directions = torch.tensor(
            get_ray_directions_blender(H, W, self.focal)
        )  # (H, W, 3)s; note that we assume all cameras share the same intrinsics
        
        if self.split == "train":
            all_times = []
            all_rays = []
            all_poses = []
            all_depths = []
            all_flows = []
            all_dymasks = []
            count = 12

            # to prevent malicious file list
            assert count == len(images)

            for index in range(0, len(images)):
                video_times = torch.tensor([index / (count - 1)]) # monocular
                all_times.append(video_times)
                pose = poses[index]
                all_poses.append(torch.tensor([pose], dtype=torch.float32))

                rays_o, rays_d = get_rays(
                    self.directions, torch.FloatTensor(poses[index])
                )  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
                all_rays += [torch.cat([rays_o, rays_d], 1)]

                disp = cv2.resize(read_MiDaS_disp(disp_paths[index], 3.0), (W, H),
                                  interpolation=cv2.INTER_NEAREST) # copied from NSFF code
                disp = np.float32(disp)
                disp = torch.FloatTensor(disp)

                if index < len(images) - 1:
                    fwd_data = np.load(fwd_paths[index])
                    fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
                    fwd_mask = np.float32(fwd_mask)
                    flow_fw = torch.FloatTensor(np.concatenate([fwd_flow, fwd_mask[...,None]], axis=-1))
                else:
                    flow_fw = torch.zeros_like(flow_fw)

                if index > 0:
                    bwd_data = np.load(bwd_paths[index - 1])
                    bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
                    bwd_mask = np.float32(bwd_mask)
                    flow_bw = torch.FloatTensor(np.concatenate([bwd_flow, bwd_mask[...,None]], axis=-1))
                else:
                    flow_bw = torch.zeros_like(flow_fw)
                
                flows = torch.concat([flow_fw, flow_bw], dim=-1)
                all_flows.append(flows)
                all_depths.append(disp)

                dymask = cv2.imread(dymask_paths[index])
                dymask = np.float32(dymask > 1e-3)[...,0]
                all_dymasks.append(torch.FloatTensor(dymask))

                print(f"video {index} is loaded")
                gc.collect()

            # load all video images
            all_imgs = [(cv2.imread(path)[:,:,[2,1,0]]/255.).astype(np.float32) for path in images]
            all_imgs = torch.tensor(all_imgs).detach()
            all_imgs = torch.reshape(all_imgs, [len(images), 1, H*W, 3])
            all_times = torch.stack(all_times, 0).detach()
            all_rays = torch.stack(all_rays, 0).detach()
            all_poses = torch.stack(all_poses, 0).detach()
            all_flows = torch.stack(all_flows, 0).detach()
            all_depths = torch.stack(all_depths, 0).detach()
            all_dymasks = torch.stack(all_dymasks, 0).detach()

            # breakpoint() # for debugging.

            print("stack performed")


            N_cam, N_time, N_rays, C = all_imgs.shape
            self.image_stride = N_rays
            self.cam_number = N_cam
            self.time_number = N_time
            
            self.frame_interval = self.time_scale * 2.0 / (count - 1)
            self.global_mean_rgb = torch.mean(all_imgs, dim=1)
            all_times = self.time_scale * (all_times * 2.0 - 1.0)

            if not self.is_stack:
                self.all_rgbs = all_imgs.view(-1, 3)
                self.all_times = all_times.view(N_cam, N_time, 1)
                self.all_rays = all_rays.view(N_cam, N_rays, 6)
                self.all_poses = all_poses.view(N_cam, N_time, 3, 4)
                self.all_flows = all_flows.view(-1, 2, 3)
                self.all_depths = all_depths.view(-1)
                self.all_dymasks = all_dymasks.view(-1)
            else:
                self.all_rgbs = all_imgs
                self.all_times = all_times.view(N_cam, N_time, 1)
                self.all_rays = all_rays.view(N_cam, N_rays, 6)
                self.all_poses = all_poses.view(N_cam, N_time, 3, 4)
                self.all_flows = all_flows.view(N_cam, N_time, N_rays, 2, 3)
                self.all_depths = all_depths.view(N_cam, N_time, N_rays)
                self.all_dymasks = all_dymasks.view(N_cam, N_time, N_rays)
            
        else:
            all_times = []
            all_rays = []
            all_poses = []
            count = 12
            camera_cnt = 1
            for camera_i in range(0, camera_cnt):
                video_times = torch.tensor([img_i / (count - 1) for img_i in range(12)])
                
                all_times.append(video_times)
                
                pose = poses[camera_i]
                all_poses.append(torch.tensor([pose] * video_times.shape[0], dtype=torch.float32))

                rays_o, rays_d = get_rays(
                    self.directions, torch.FloatTensor(poses[camera_i])
                )  # both (h*w, 3)

                rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
                all_rays += [torch.cat([rays_o, rays_d], 1)]
                
                gc.collect()
            
            all_times = torch.stack(all_times, 0).detach()
            all_rays = torch.stack(all_rays, 0).detach()
            all_poses = torch.stack(all_poses).detach()
            
            all_imgs = []
            all_paths = []

            for camera_i in range(0, camera_cnt):
                one_video = []
                for img_i in range(12):
                    gt_img_path = os.path.join(self.root_dir, 
                                               'test', 
                                               'v000_t%03d.png' % img_i)
                    
                    gt_img = cv2.imread(gt_img_path)[:, :, ::-1] / 255.
                    gt_img = cv2.resize(gt_img, 
                                        dsize=(W, H), 
                                        interpolation=cv2.INTER_AREA)
                    
                    one_video.append(torch.tensor(gt_img.astype(np.float32)))
                    all_paths.append(gt_img_path)
                print(f"video {camera_i} is loaded")
                one_video = torch.stack(one_video, 0)
                all_imgs.append(one_video)
            
            all_imgs = torch.stack(all_imgs, 0)
            all_imgs = torch.reshape(all_imgs, [camera_cnt, 12, H*W, 3])

            gc.collect()
            N_cam, N_time, N_rays, C = all_imgs.shape
            self.image_stride = N_rays
            self.time_number = N_time
            self.all_rgbs = all_imgs.view(-1, N_rays, 3)
            self.all_rays = all_rays
            self.all_rgbs = self.all_rgbs.view(
                camera_cnt, 12, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = self.time_scale * (all_times * 2.0 - 1.0)
            self.frame_interval = self.time_scale * 2.0 / (count - 1)
            self.all_poses = all_poses
            self.all_paths = all_paths

    def __len__(self):
        if self.split == "train" and self.is_stack is True:
            return self.cam_number * self.time_number
        elif self.split != "train":
            return self.all_rgbs.shape[0] * self.all_rgbs.shape[1]
        else:
            return self.all_rgbs.shape[0]

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            if self.is_stack:
                if not isinstance(idx, int) and len(idx) == 1:
                    idx = idx[0]
                cam_idx = idx // self.time_number
                time_idx = idx % self.time_number

                prev_cam_idx = np.maximum(cam_idx - 1, 0)
                prev_time_idx = 0
                post_cam_idx = np.minimum(cam_idx + 1, self.cam_number- 1)
                post_time_idx = np.minimum(time_idx + 1, self.time_number - 1)

                sample = {
                    "rays": self.all_rays[cam_idx],
                    "rgbs": self.all_rgbs[cam_idx, time_idx],
                    "depths": self.all_depths[cam_idx, time_idx],
                    "dymasks": self.all_dymasks[cam_idx, time_idx],
                    "time": self.all_times[cam_idx, time_idx]
                    * torch.ones_like(self.all_rays[cam_idx][:, 0:1]),
                    "pose": self.all_poses[cam_idx, time_idx] 
                    * torch.ones_like(self.all_rays[cam_idx][:, 0:1][...,None]),
                    "prev_pose": self.all_poses[prev_cam_idx, prev_time_idx] 
                    * torch.ones_like(self.all_rays[prev_cam_idx][:, 0:1][...,None]),
                    "post_pose": self.all_poses[post_cam_idx, post_time_idx] 
                    * torch.ones_like(self.all_rays[post_cam_idx][:, 0:1][...,None]),
                    "flows": self.all_flows[cam_idx, time_idx],
                }

            else:
                idx_prev = np.maximum(idx - self.time_number * self.image_stride, 0)
                idx_post = np.minimum(idx + self.time_number * self.image_stride, self.cam_number * self.time_number * self.image_stride - 1)
                sample = {
                    "rays": self.all_rays[
                        idx // (self.time_number * self.image_stride),
                        idx % (self.image_stride),
                    ],
                    "rgbs": self.all_rgbs[idx],
                    "depths": self.all_depths[idx],
                    "dymasks": self.all_dymasks[idx],
                    "time": self.all_times[
                        idx // (self.time_number * self.image_stride),
                        idx
                        % (self.time_number * self.image_stride)
                        // self.image_stride,
                    ]
                    * torch.ones_like(self.all_rgbs[idx][:, 0:1]),
                    "pose": self.all_poses[idx // (self.time_number * self.image_stride),
                                            idx
                                            % (self.time_number * self.image_stride)
                                            // self.image_stride]
                    * torch.ones_like(self.all_rgbs[idx][:, 0:1][...,None]),
                    "prev_pose": self.all_poses[idx_prev // (self.time_number * self.image_stride),
                                            idx_prev
                                            % (self.time_number * self.image_stride)
                                            // self.image_stride]
                    * torch.ones_like(self.all_rgbs[idx_prev][:, 0:1][...,None]),
                    "post_pose": self.all_poses[idx_post // (self.time_number * self.image_stride),
                                            idx_post
                                            % (self.time_number * self.image_stride)
                                            // self.image_stride]
                    * torch.ones_like(self.all_rgbs[idx_post][:, 0:1][...,None]),
                    "flows": self.all_flows[idx],
                }

        else:  # create data for each image separately
            cam_idx = idx // self.time_number
            time_idx = idx % self.time_number
            if self.is_stack:
                sample = {
                    "rays": self.all_rays[cam_idx],
                    "rgbs": self.all_rgbs[cam_idx, time_idx],
                    "time": self.all_times[cam_idx, time_idx]
                    * torch.ones_like(self.all_rays[cam_idx][:, 0:1]),
                    "pose": self.all_poses[cam_idx, time_idx] 
                    * torch.ones_like(self.all_rays[cam_idx][:, 0:1][...,None]),
                }

            else:
                sample = {
                    "rays": self.all_rays[
                        idx // (self.time_number * self.image_stride),
                        idx % (self.image_stride),
                    ],
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[
                        idx // (self.time_number * self.image_stride),
                        idx
                        % (self.time_number * self.image_stride)
                        // self.image_stride,
                    ]
                    * torch.ones_like(self.all_rgbs[idx][:, 0:1]),
                    "pose": self.all_poses[idx // (self.time_number * self.image_stride),
                                            idx
                                            % (self.time_number * self.image_stride)
                                            // self.image_stride]
                    * torch.ones_like(self.all_rgbs[idx][:, 0:1][...,None]),
                }

        return sample

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                W, H = self.img_wh
                rays_o, rays_d = ndc_rays_blender(
                    H, W, self.focal[0], 1.0, rays_o, rays_d
                )
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

