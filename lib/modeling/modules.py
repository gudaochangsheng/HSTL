import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, d, h, w]
            outs: [n, c, d, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False