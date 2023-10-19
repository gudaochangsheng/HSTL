import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, BasicConv2d

# frame-level temporal aggregation (FTA)
# frame-level temporal aggregation (FTA)
class FTA(nn.Module):

    def __init__(self, channel=64, FP_kernels=[3, 5]):
        super().__init__()
        self.FP_3 = nn.MaxPool3d(kernel_size=(FP_kernels[0],1,1),stride=(3,1,1),padding=(0,0,0))
        self.FP_5 = nn.MaxPool3d(kernel_size=(FP_kernels[1],1,1),stride=(3,1,1),padding=(1,0,0))
        self.gap = nn.AdaptiveAvgPool3d((None,1,1))
        self.fcs = nn.ModuleList([])
        for i in range(len(FP_kernels)):
            self.fcs.append(nn.Conv1d(channel,channel,1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, t,_, _ = x.size()
        aggregate_outs = []
        outs1 = self.FP_3(x)
        aggregate_outs.append(outs1)
        outs2 = self.FP_5(x)
        aggregate_outs.append(outs2)
        aggregate_features = torch.stack(aggregate_outs, 0)
        hat_out_mid = sum(aggregate_outs)
        hat_out = self.gap(hat_out_mid).squeeze(-1).squeeze(-1)
        temporal = hat_out.size(-1)
        weights = []
        for fc in self.fcs:
            weight = fc(hat_out)
            weights.append(weight.view(bs, c, temporal, 1, 1))
        select_weights = torch.stack(weights, 0)
        select_weights = self.softmax(select_weights)
        outs = (select_weights * aggregate_features).sum(0)
        return outs

class FTA_Block(nn.Module):
    def __init__(self, split_param, m, in_channels):
        super(FTA_Block, self).__init__()
        self.split_param = split_param
        self.m = m
        self.mma = nn.ModuleList([
            FTA(channel=in_channels, FP_kernels=[3, 5])
            for i in range(self.m)])
    def forward(self, x):
        feat = x.split(self.split_param, 3)
        feat = torch.cat([self.mma[i](_) for i, _ in enumerate(feat)], 3)
        return feat

# adaptive region-based motion extractor (ARME)
class ARME_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, split_param ,m, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 padding=(1, 1, 1),bias=False,**kwargs):
        super(ARME_Conv, self).__init__()
        self.m = m

        self.split_param = split_param

        self.conv3d = nn.ModuleList([
            BasicConv3d(in_channels, out_channels, kernel_size, stride, padding,bias ,**kwargs)
            for i in range(self.m)])


    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        feat = x.split(self.split_param, 3)
        feat = torch.cat([self.conv3d[i](_) for i, _ in enumerate(feat)], 3)
        feat = F.leaky_relu(feat)
        return feat

# Generalized Mean Pooling (GeM)
class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


# adaptive spatio-temporal pooling
class ASTP(nn.Module):
    def __init__(self, split_param, m, in_channels, out_channels, flag=True):
        super(ASTP, self).__init__()
        self.split_param = split_param
        self.m = m
        self.hpp = nn.ModuleList([
            GeMHPP(bin_num=[1]) for i in range(self.m)])

        self.flag = flag
        if self.flag:
            self.proj = BasicConv2d(in_channels, out_channels, 1, 1, 0)


        self.SP1 = PackSequenceWrapper(torch.max)
    def forward(self, x, seqL):
        x = self.SP1(x, seqL=seqL, options={"dim": 2})[0]
        if self.flag:
            x = self.proj(x)
        feat = x.split(self.split_param, 2)
        feat = torch.cat([self.hpp[i](_) for i, _ in enumerate(feat)], -1)
        return feat

class HSTL(BaseModel):
    """
        Hierarchical Spatio-Temporal Feature Learning for Gait Recognition
    """

    def __init__(self, *args, **kargs):
        super(HSTL, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        # For CASIA-B dataset.
        self.arme1 = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.astp1 = ASTP(split_param=[64], m=1, in_channels=in_c[0], out_channels=in_c[-1])

        self.arme2 = nn.Sequential(
            ARME_Conv(in_c[0], in_c[0], split_param=[40, 24], m=2, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ARME_Conv(in_c[0], in_c[1], split_param=[40, 24], m=2, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )

        self.astp2 = ASTP(split_param=[40, 24], m=2, in_channels=in_c[1], out_channels=in_c[-1])

        self.fta = FTA_Block(split_param=[40, 24], m=2, in_channels=in_c[1])

        self.astp2_fta = ASTP(split_param=[40, 24], m=2, in_channels=in_c[1], out_channels=in_c[-1])

        self.arme3 = nn.Sequential(
            ARME_Conv(in_c[1], in_c[2], split_param=[8, 32, 16, 8], m=4, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ARME_Conv(in_c[2], in_c[2], split_param=[8, 32, 16, 8], m=4, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )

        self.astp3 = ASTP(split_param=[8, 32, 16, 8], m=4, in_channels=in_c[2], out_channels=in_c[-1], flag=False)

        # self.astp4 = ASTP(split_param=[1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1,
        #                                1,1,1,1,1,1,1,1], m=64, in_channels=in_c[2], out_channels=in_c[-1])
        self.HPP = GeMHPP()

        # separable fully connected layer (SeFC)
        self.Head0 = SeparateFCs(73, in_c[-1], in_c[-1])
        # batchnorm layer (BN)
        self.Bn = nn.BatchNorm1d(in_c[-1])
        # separable fully connected layer (SeFC)
        self.Head1 = SeparateFCs(73, in_c[-1], class_num)
        # Temporal Pooling (TP)
        self.TP = PackSequenceWrapper(torch.max)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)
        outs = self.arme1(sils)
        astp1 = self.astp1(outs, seqL)
        outs = self.arme2(outs)
        astp2 = self.astp2(outs, seqL)
        outs = self.fta(outs)
        astp2_fta = self.astp2_fta(outs, seqL)
        outs = self.arme3(outs)
        astp3 = self.astp3(outs, seqL)
        astp4 = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        astp4 = self.HPP(astp4)
        # astp4 = self.astp4(outs, seqL)
        outs = torch.cat([astp1,astp2, astp2_fta, astp3, astp4], dim=-1) # [n, c, p]
        outs = outs.permute(2, 0, 1).contiguous()  # [p, n, c]
        gait = self.Head0(outs)  # [p, n, c]
        gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        bnft = self.Bn(gait)  # [n, c, p]
        logi = self.Head1(bnft.permute(2, 0, 1).contiguous())  # [p, n, c]

        gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]
        bnft = bnft.permute(0, 2, 1).contiguous()  # [n, p, c]
        logi = logi.permute(1, 0, 2).contiguous()  # [n, p, c]
        # print(logi.size())

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': bnft, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': bnft
            }
        }
        return retval
