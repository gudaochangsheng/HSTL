from ctypes import ArgumentError
import torch.nn as nn
import torch
from utils import Odict
import functools
from utils import ddp_all_gather


def gather_and_scale_wrapper(func):

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:

            for k, v in kwds.items():
                kwds[k] = ddp_all_gather(v)

            loss, loss_info = func(*args, **kwds)
            loss *= torch.distributed.get_world_size()
            return loss, loss_info
        except:
            raise ArgumentError
    return inner


class BaseLoss(nn.Module):

    def __init__(self, loss_term_weight=1.0):

        super(BaseLoss, self).__init__()
        self.loss_term_weight = loss_term_weight
        self.info = Odict()

    def forward(self, logits, labels):

        return .0, self.info
