import torch
from torch import nn
from networks.guide_filter import GuidedFilter
import torch.nn.functional as F


class GFN(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, dgf_r=4, dgf_eps=1e-2):
        super(GFN, self).__init__()
        self.guided_map_conv1 = nn.Conv3d(in_channels, 64, 1)
        self.guided_map_relu1 = nn.ReLU(inplace=True)
        self.guided_map_conv2 = nn.Conv3d(64, out_channels, 1)
        self.guided_filter = GuidedFilter(dgf_r, dgf_eps)

    def forward(self, input, out):
        g = self.guided_map_relu1(self.guided_map_conv1(input))
        g = self.guided_map_conv2(g)
        out = self.guided_filter(g, out)

        return out