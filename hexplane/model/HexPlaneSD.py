import torch
from torch.nn import functional as F

from hexplane.model.HexPlaneSD_Base import HexPlaneSD_Base


class HexPlaneSD(HexPlaneSD_Base):
    """
    A SD version of HexPlane, which supports different fusion methods and feature regressor methods.
    """

    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kargs):
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kargs)