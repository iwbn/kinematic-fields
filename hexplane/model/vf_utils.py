import numpy as np
import torch
import torch.nn.functional as F


def world2ndc(pts, H, W, focal, near, epsilon=1e-6, **kwargs):
    x = pts[...,0]
    y = pts[...,1]
    z = pts[...,2] + epsilon

    ax = -1./(W/(2.*focal))
    ay = -1./(H/(2.*focal))
    
    x_ndc = x * ax / z
    y_ndc = y * ay / z
    z_ndc = 1. + 2. * near / z

    if isinstance(x_ndc, torch.Tensor):
        ndc = torch.stack([x_ndc, y_ndc, z_ndc], dim=-1)
    else:
        ndc = np.stack([x_ndc, y_ndc, z_ndc], axis=-1)
    return ndc


def jacobian_world2ndc(pts_ndc, H, W, focal, near, epsilon=1e-6):
    x_ndc = pts_ndc[...,0]
    y_ndc = pts_ndc[...,1]
    z_ndc = pts_ndc[...,2]

    ax = -1./(W/(2.*focal))
    ay = -1./(H/(2.*focal))

    z = (2. * near) / (z_ndc - 1. + epsilon)
    x = z * x_ndc / ax
    y = z * y_ndc / ay


    if isinstance(x_ndc, torch.Tensor):
        zero = torch.zeros_like(x_ndc)
        J = torch.stack([
            torch.stack([ax / z, zero, -x*ax / (z*z)], dim = -1),
            torch.stack([zero, ay / z, -y*ay / (z*z)], dim = -1),
            torch.stack([zero, zero, -2.*near / (z*z)], dim = -1)
        ], -2)
    else:
        zero = np.zeros_like(x_ndc)
        J = np.stack([
            np.stack([ax / z, zero, -x*ax / (z*z)], axis = -1),
            np.stack([zero, ay / z, -y*ay / (z*z)], axis = -1),
            np.stack([zero, zero, -2.*near / (z*z)], axis = -1)
        ], -2)

    J = J 
    return J


def ndc2world(pts_ndc, H, W, focal, near, epsilon=1e-6, **kwargs):
    x_ndc = pts_ndc[...,0]
    y_ndc = pts_ndc[...,1]
    z_ndc = pts_ndc[...,2]

    ax = -1./(W/(2.*focal))
    ay = -1./(H/(2.*focal))
    
    z = (2. * near) / (z_ndc - 1. + epsilon)
    x = z * x_ndc / ax
    y = z * y_ndc / ay

    if isinstance(x_ndc, torch.Tensor):
        world = torch.stack([x, y, z], dim=-1)
    else:
        world = np.stack([x, y, z], axis=-1)
    return world


def ndc2world_sigma(sigma, pts_ndc, H, W, focal, near, epsilon=1e-6):
    J = jacobian_world2ndc(pts_ndc, H, W, focal, near, epsilon)
    if isinstance(pts_ndc, torch.Tensor):
        scale = torch.abs(torch.linalg.det(J.detach()))
    else:
        scale = np.abs(np.linalg.det(J))
    sigma_world = sigma * scale

    return sigma_world


def get_strain_rate_grad(grad_x, grad_y, grad_z, ndc_coords, ndc_params):
    if "H" in ndc_params:
        H = ndc_params['H']
        W = ndc_params['W']
        focal = ndc_params['focal']
        near = ndc_params['near']
    
    if ndc_params['use_ndc']:
        dV_dNdc = torch.stack([grad_x, grad_y, grad_z], dim=-1)
        sNdc_dX = jacobian_world2ndc(ndc_coords, H, W, focal, near)
        dV_dX = torch.matmul(dV_dNdc, sNdc_dX)
    else:
        dV_dX = torch.stack([grad_x, grad_y, grad_z], dim=-1)

    strain_rate_tensor = 0.5 * (dV_dX + torch.transpose(dV_dX, -2, -1))
    t1 = torch.einsum("...ii", strain_rate_tensor) ** 2.0
    t2 = torch.einsum("...ii", torch.matmul(strain_rate_tensor, strain_rate_tensor))
    t = 0.5 * (t1 - t2)
    res = t
    return res


def get_higher_order(velocity, frame_time, ndc_params):
    if "H" in ndc_params:
        H = ndc_params['H']
        W = ndc_params['W']
        focal = ndc_params['focal']
        near = ndc_params['near']

    v_x = velocity[..., 0]
    v_y = velocity[..., 1]
    v_z = velocity[..., 2]

    grad_x = torch.autograd.grad(v_x.sum(), frame_time, create_graph=True)[0][...,:3]
    grad_y = torch.autograd.grad(v_y.sum(), frame_time, create_graph=True)[0][...,:3]
    grad_z = torch.autograd.grad(v_z.sum(), frame_time, create_graph=True)[0][...,:3]
    
    dV_dt = torch.cat([grad_x, grad_y, grad_z], dim=-1)

    return dV_dt
    

def get_strain_rate_world(velocity, world_coords):

    v_x = velocity[..., 0]
    v_y = velocity[..., 1]
    v_z = velocity[..., 2]

    grad_x = torch.autograd.grad(v_x.sum(), world_coords, create_graph=True)[0]
    grad_y = torch.autograd.grad(v_y.sum(), world_coords, create_graph=True)[0]
    grad_z = torch.autograd.grad(v_z.sum(), world_coords, create_graph=True)[0]

    dV_dX = torch.stack([grad_x, grad_y, grad_z], dim=-2)

    strain_rate_tensor = 0.5 * (dV_dX + torch.transpose(dV_dX, -2, -1))
    a = torch.abs(strain_rate_tensor)

    return a


def taylor_order_val(x, order):
    if not torch.is_tensor(x):
        res = np.power(x, order) / np.math.factorial(order)
    else:
        res = torch.pow(x, order) / np.math.factorial(order)
    return res


def get_disp_at(x, flow_order, raw):
    if isinstance(raw, torch.Tensor):
        raw_sf_at_x = torch.zeros_like(raw[..., 0:3])
    else:
        raw_sf_at_x = np.zeros_like(raw[..., 0:3])

    for order in range(flow_order):
        diff_0 = raw[..., 3 * order:3 * (order+1)] 
        raw_sf_at_x += diff_0 * taylor_order_val(x, order + 1)
    return raw_sf_at_x


def get_new_coords(coords, flow, ndc_params=None, epsilon=1e-6):
    if ndc_params is not None:
        H = ndc_params['H']
        W = ndc_params['W']
        focal = ndc_params['focal']
        near = ndc_params['near']

        new_coords_world = ndc2world(coords, H, W, focal, near, epsilon) + flow
        new_coords = world2ndc(new_coords_world, H, W, focal, near, epsilon)
    else:
        new_coords = coords + flow

    return new_coords


def OctreeRender_trilinear_scene_flow(
    rays,
    time,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    poses=None,
    device="cuda",
    **kwargs
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals = [], [], [], []
    res_sf_dict = {}
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        poses_chunk = None
        if poses is not None:
            poses_chunk = {}
            for k, v in poses.items():
                poses_chunk[k] = v[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map, alpha_map, z_val_map, sf_res = model(
            rays_chunk,
            time_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
            poses=poses_chunk,
            **kwargs
        )
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)

        if sf_res is not None:
            for k, v in sf_res.items():
                res_sf_dict.setdefault(k, []).append(v)
                
    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        res_sf_dict,
    )

"""
CODE BLOCK FROM NSFF BEGIN
"""

def projection_from_ndc(c2w, H, W, f, weights_ref, raw_pts, n_dim=1):
    R_w2c = c2w[..., :3, :3].transpose(0, 1)
    t_w2c = -torch.matmul(R_w2c, c2w[..., :3, 3:])

    pts_3d = torch.sum(weights_ref[..., None] * raw_pts, -2)  # [N_rays, 3]

    pts_3d_e_world = NDC2Euclidean(pts_3d, H, W, f)

    if n_dim == 1:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(-3), 
                                              t_w2c.unsqueeze(-3))
    else:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(-3).unsqueeze(-3), 
                                              t_w2c.unsqueeze(-3).unsqueeze(-3))

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f)

    return pts_2d


def NDC2Euclidean(xyz_ndc, H, W, f):
    z_e = 2./ (xyz_ndc[..., 2:3] - 1. + 1e-6)
    x_e = - xyz_ndc[..., 0:1] * z_e * W/ (2. * f)
    y_e = - xyz_ndc[..., 1:2] * z_e * H/ (2. * f)

    xyz_e = torch.cat([x_e, y_e, z_e], -1)
 
    return xyz_e


"CODE BLOCK FROM NSFF ENDS"


def compute_sf_sm(scene_flow_world):
    return torch.abs(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])


def se3_transform_points(pts_ref, raw_rot_ref2prev, raw_trans_ref2prev):
    pts_prev = torch.squeeze(torch.matmul(raw_rot_ref2prev, pts_ref[..., :3].unsqueeze(-1)) + raw_trans_ref2prev)
    return pts_prev


def compute_optical_flow(c2w_ref, c2w, H, W, f, dyweight, stweight, orig_pts, raw_pts, center=None):
    R_w2c = c2w[..., :3, :3].transpose(-2, -1) # [N_rays, 3]
    t_w2c = -torch.matmul(R_w2c, c2w[..., :3, 3:])

    R_w2c_ref = c2w_ref[..., :3, :3].transpose(-2, -1) # [N_rays, 3]
    t_w2c_ref = -torch.matmul(R_w2c_ref, c2w_ref[..., :3, 3:])

    weight_blended = dyweight + stweight
    pts_3d_blended = dyweight[...,None] * raw_pts + stweight[...,None] * orig_pts

    acc_blended = torch.sum(weight_blended, dim=-1)
    pts_3d = torch.sum(pts_3d_blended, -2)
    pts_3d += (1.-acc_blended)[...,None] * orig_pts[...,-1,:]

    pts_3d = ndc2world(pts_3d, H, W, f, 1.0, epsilon=0.)
    orig_pts = ndc2world(orig_pts[...,1,:], H, W, f, 1.0, epsilon=0.) # [N_rays, 3]

    pts_3d_e_local = se3_transform_points(pts_3d, R_w2c, t_w2c)
    pts_3d_e_orig = se3_transform_points(orig_pts, R_w2c_ref, t_w2c_ref)

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f, center=center)
    pts_2d_orig = perspective_projection(pts_3d_e_orig, H, W, f, center=center)

    return pts_2d - pts_2d_orig


def compute_optical_flow_world(c2w_ref, c2w, H, W, f, dyweight, stweight, orig_pts, raw_pts, center=None):
    R_w2c = c2w[..., :3, :3].transpose(-2, -1) # [N_rays, 3]
    t_w2c = -torch.matmul(R_w2c, c2w[..., :3, 3:])

    R_w2c_ref = c2w_ref[..., :3, :3].transpose(-2, -1) # [N_rays, 3]
    t_w2c_ref = -torch.matmul(R_w2c_ref, c2w_ref[..., :3, 3:])

    weight_blended = dyweight + stweight
    pts_3d_blended = dyweight[...,None] * raw_pts + stweight[...,None] * orig_pts

    acc_blended = torch.sum(weight_blended, dim=-1)
    pts_3d = torch.sum(pts_3d_blended, -2)
    pts_3d += (1.-acc_blended)[...,None] * orig_pts[...,-1,:]

    orig_pts = orig_pts[...,1,:]

    pts_3d_e_local = se3_transform_points(pts_3d, R_w2c, t_w2c)
    pts_3d_e_orig = se3_transform_points(orig_pts, R_w2c_ref, t_w2c_ref)

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f, center=center)
    pts_2d_orig = perspective_projection(pts_3d_e_orig, H, W, f, center=center)

    return pts_2d - pts_2d_orig


# NOTE: WE DO IN COLMAP/OPENCV FORMAT, BUT INPUT IS OPENGL FORMAT!!!!!1
def perspective_projection(pts_3d, h, w, f, center=None):
    if center is None:
        center = (w/2., h/2.)
    pts_2d = torch.cat([pts_3d[..., 0:1] * f/-pts_3d[..., 2:3] + center[0], 
                        -pts_3d[..., 1:2] * f/-pts_3d[..., 2:3] + center[1]], dim=-1)

    return pts_2d


def compute_depth_loss(pred_depth, gt_depth, mask=None, frame_time=None, frame_interval=None):
    if mask is None:
        mask = torch.ones_like(pred_depth)

    pred_depth_valid = pred_depth
    gt_depth_valid = gt_depth
    
    t_pred = torch.median(pred_depth_valid)
    s_pred = torch.mean(torch.abs(pred_depth_valid - t_pred))

    t_gt = torch.median(gt_depth_valid)
    s_gt = torch.mean(torch.abs(gt_depth_valid - t_gt))

    pred_depth_n = (pred_depth_valid - t_pred)/s_pred
    gt_depth_n = (gt_depth_valid - t_gt)/s_gt

    loss = torch.mean(mask * (pred_depth_n - gt_depth_n)**2)
    return loss


class ExponentialMovingAverage:
    def __init__(self, alpha):
        self.alpha = alpha
        self.average = None

    def update(self, value):
        if self.average is None:
            self.average = value
        else:
            self.average = self.alpha * self.average + (1.0 - self.alpha) * value


def chabonnier(diff, epsilon=1e-4):
    inner = diff ** 2 + epsilon ** 2
    res = inner ** 0.5
    return res


def compute_higher_field(grad_x, grad_y, grad_z, grad_t, v):
    del_v = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    v_del_v = v[...,None,:] * del_v
    v_del_v = v_del_v.sum(dim=-1)
    a = grad_t + v_del_v

    return a


def sample_spherical(num_samples):
    u = torch.rand(num_samples)
    v = torch.rand(num_samples)
    
    theta = 2 * torch.pi * u
    phi = torch.acos(2 * v - 1)
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=-1)
