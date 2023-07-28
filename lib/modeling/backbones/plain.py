import torch.nn as nn
from ..modules import BasicConv2d


class Plain(nn.Module):

    def __init__(self, layers_cfg, in_channels=1):
        super(Plain, self).__init__()
        self.layers_cfg = layers_cfg
        self.in_channels = in_channels

        self.feature = self.make_layers()

    def forward(self, seqs):
        out = self.feature(seqs)
        return out

    def make_layers(self):

        def get_layer(cfg, in_c, kernel_size, stride, padding):
            cfg = cfg.split('-')
            typ = cfg[0]
            if typ not in ['BC', 'FC']:
                raise ValueError('Only support BC or FC, but got {}'.format(typ))
            out_c = int(cfg[1])

            if typ == 'BC':
                return BasicConv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)

        Layers = [get_layer(self.layers_cfg[0], self.in_channels,
                            5, 1, 2), nn.LeakyReLU(inplace=True)]
        in_c = int(self.layers_cfg[0].split('-')[1])
        for cfg in self.layers_cfg[1:]:
            if cfg == 'M':
                Layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = get_layer(cfg, in_c, 3, 1, 1)
                Layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_c = int(cfg.split('-')[1])
        return nn.Sequential(*Layers)