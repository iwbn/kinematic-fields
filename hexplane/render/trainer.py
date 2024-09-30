import os
import math
import sys
import glob
from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm

from hexplane.render.util.Sampling import cal_n_samples
from hexplane.render.util.util import N_to_reso

LOG_DETAILS=False

# frequently used function
_c = lambda x: torch.cat(x, dim=0)

class SimpleSampler:
    """
    A sampler that samples a batch of ids randomly.
    """

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class Trainer:
    def __init__(
        self,
        model,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    ):
        self.model = model
        self.cfg = cfg
        self.reso_cur = reso_cur
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.summary_writer = summary_writer
        self.logfolder = logfolder
        self.device = device

    def get_lr_decay_factor(self, step):
        """
        Calculate the learning rate decay factor = current_lr / initial_lr.
        """
        if self.cfg.optim.lr_decay_step == -1:
            self.cfg.optim.lr_decay_step = self.cfg.optim.n_iters

        if self.cfg.optim.lr_decay_type == "exp":  # exponential decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio ** (
                step / self.cfg.optim.lr_decay_step
            )
        elif self.cfg.optim.lr_decay_type == "linear":  # linear decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio + (
                1 - self.cfg.optim.lr_decay_target_ratio
            ) * (1 - step / self.cfg.optim.lr_decay_step)
        elif self.cfg.optim.lr_decay_type == "cosine":  # consine decay
            lr_factor = self.cfg.optim.lr_decay_target_ratio + (
                1 - self.cfg.optim.lr_decay_target_ratio
            ) * 0.5 * (1 + math.cos(math.pi * step / self.cfg.optim.lr_decay_step))

        return lr_factor

    def get_voxel_upsample_list(self):
        """
        Precompute  spatial and temporal grid upsampling sizes.
        """
        upsample_list = self.cfg.model.upsample_list
        if (
            self.cfg.model.upsampling_type == "unaligned"
        ):  # logaritmic upsampling. See explanation of "unaligned" in model/__init__.py.
            N_voxel_list = (
                torch.round(
                    torch.exp(
                        torch.linspace(
                            np.log(self.cfg.model.N_voxel_init),
                            np.log(self.cfg.model.N_voxel_final),
                            len(upsample_list) + 1,
                        )
                    )
                ).long()
            ).tolist()[1:]
        elif (
            self.cfg.model.upsampling_type == "aligned"
        ):  # aligned upsampling doesn't need precompute N_voxel_list.
            N_voxel_list = None
        # logaritmic upsampling for time grid.
        Time_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.cfg.model.time_grid_init),
                        np.log(self.cfg.model.time_grid_final),
                        len(upsample_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
        self.N_voxel_list = N_voxel_list
        self.Time_grid_list = Time_grid_list
    
    def sample_data(self, train_dataset, iteration):
        """
        Sample a batch of data from the dataset.
        """
        raise NotImplementedError

    def init_sampler(self, train_dataset):
        """
        Initialize the sampler for the training dataset.
        """
        if self.cfg.data.datasampler_type == "rays":
            self.sampler = SimpleSampler(len(train_dataset), self.cfg.optim.batch_size)
        elif self.cfg.data.datasampler_type == "images":
            self.sampler = SimpleSampler(len(train_dataset), 1)
        
    # main train function
    def train(self):
        raise NotImplementedError

    def get_optimizer(self, hexplane, iteration):
        raise NotImplementedError
    
    def save_hexplane(self, hexplane, iteration, reso_cur, nSamples, optimizer, flow_optimizer):
        tosave = {'iteration': iteration,
                    'reso_cur': reso_cur,
                    'N_voxel_list': self.N_voxel_list,
                    'Time_grid_list': self.Time_grid_list,
                    'nSamples': nSamples,
                    'hexplane_state_dict': hexplane.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }

        if 'static_model' in hexplane._modules:
             tosave['static_model'] = hexplane.static_model

        if flow_optimizer is not None:
            tosave['flow_optimizer_state_dict'] = flow_optimizer.state_dict()

        if self.cfg.light_mode:
            to_remove = glob.glob(f"{self.logfolder}/{self.cfg.expname}_*.th")
            for f in to_remove:
                os.remove(f)
                
        torch.save(tosave, 
        f"{self.logfolder}/{self.cfg.expname}_{iteration}.th")

    def load_hexplane(self, ckpt_path, hexplane, optimizer, flow_optimizer, reso_cur, nSamples):
        ckpt = torch.load(ckpt_path)
        start = ckpt['iteration'] + 1

        for inner_iter in range(start): # update resolutions using the virtual training loop below
            optimizer, flow_optimizer, \
                reso_cur, nSamples = self.update_bunch(hexplane, optimizer, flow_optimizer, reso_cur, nSamples, inner_iter)
        
        if "static_model" in hexplane._modules:
            hexplane.static_model = ckpt['static_model']
            hexplane.update_static_model_params()
        hexplane.load_state_dict(ckpt['hexplane_state_dict'], strict=False)
        
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if flow_optimizer is not None:
            flow_optimizer.load_state_dict(ckpt['flow_optimizer_state_dict'])

        assert self.N_voxel_list == ckpt['N_voxel_list']
        assert self.Time_grid_list == ckpt['Time_grid_list']

        optimizer, flow_optimizer, \
            reso_cur, nSamples = self.update_bunch(hexplane, optimizer, flow_optimizer, reso_cur, nSamples, start-1)
        
        return hexplane, optimizer, flow_optimizer, reso_cur, nSamples, start
        
    def update_bunch(self, hexplane, optimizer, flow_optimizer, reso_cur, nSamples, iteration):
        # Calculate the emptiness voxel.
        if iteration in self.cfg.model.update_emptymask_list:
            if (
                reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
            ):  # update volume resolution
                reso_mask = reso_cur
                hexplane.updateEmptyMask(tuple(reso_mask))

        # Upsample the volume grid.
        if iteration in self.cfg.model.upsample_list:
            idx = self.cfg.model.upsample_list.index(iteration)
            if self.cfg.model.upsampling_type == "aligned":
                reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
            else:
                N_voxel = self.N_voxel_list.pop(0)
                reso_cur = N_to_reso(
                    N_voxel, hexplane.aabb, self.cfg.model.nonsquare_voxel
                )
            time_grid = self.Time_grid_list.pop(0)
            nSamples = min(
                self.cfg.model.nSamples,
                cal_n_samples(reso_cur, self.cfg.model.step_ratio),
            )
            hexplane.upsample_volume_grid(reso_cur, time_grid)

            init_timeinter = self.cfg.model.init_timeinter
            max_timeinter = self.cfg.model.max_timeinter
            init_timehop = self.cfg.model.init_timehop
            max_timehop = self.cfg.model.max_timehop
            timeinter_list = list(np.linspace(init_timeinter, max_timeinter, len(self.cfg.model.upsample_list) + 1))
            timehop_list = list(np.linspace(init_timehop, max_timehop, len(self.cfg.model.upsample_list) + 1))
            hexplane.max_timeinter = timeinter_list[idx + 1]
            hexplane.max_timehop = timehop_list[idx + 1]
            print("MAX TIMEINTER: %.4f" % hexplane.max_timeinter)
            print("MAX TIMEHOP: %.4f" % hexplane.max_timehop)

            if optimizer is not None:
                optimizer, flow_optimizer = self.get_optimizer(hexplane, iteration)

        elif iteration == self.cfg.optim.sf_begin:
            if optimizer is not None:
                optimizer, flow_optimizer = self.get_optimizer(hexplane, iteration)
        
        elif iteration == self.cfg.optim.dsj_begin:
            if optimizer is not None:
                optimizer, flow_optimizer = self.get_optimizer(hexplane, iteration)

        if type(hexplane).__name__.startswith("HexPlaneSD_Flow") and \
            iteration in self.cfg.model.taylor_mask_unlock_list:
            hexplane.add_one_taylor_order()

        return optimizer, flow_optimizer, reso_cur, nSamples

class StaticTrainer(Trainer):
    def sample_data(self, train_dataset, iteration):
        """
        Sample a batch of data from the dataset, but only static pixels.
        """
        raise NotImplementedError

    def train(self):
        # load the training and testing dataset and other settings.
        raise NotImplementedError