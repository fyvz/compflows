
#%%
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F, init

from base import Flow,Composite
from compressai.models.utils import conv

from torch.nn.utils.parametrizations import orthogonal
class Permute(Flow):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode="shuffle"):
        """
        Constructor
        :param num_channel: Number of channels
        :param mode: Mode of permuting features, can be shuffle for
        random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == "shuffle":
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(self.num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z):
        if self.mode == "shuffle":
            z = z[:, self.perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        #log_det = 0
        return z

    def inverse(self, z):
        if self.mode == "shuffle":
            z = z[:, self.inv_perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : (self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        #log_det = 0
        return z
class OrthogonalTransform(Flow):
    # A simple Affine Flow with orthogonal matrix - 0 log det.
    def __init__(self,in_channels = 192):
        super().__init__()

        self.orth_matrix = orthogonal(nn.Linear(in_channels,in_channels,bias=None))
        self.bias        = nn.parameter.Parameter(torch.randn(in_channels))

        self.shuffle = Permute(in_channels,mode="swap")
    def internal_transform(self,x):
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x_out = x @ self.orth_matrix.weight
    def forward(self,z:torch.Tensor):
        z1 =   z.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        z2 =   z1 @ self.orth_matrix.weight.T + self.bias
        z3 =   z2.permute(0, 3, 1, 2)
        return z3

    def inverse(self, z:torch.Tensor):
        z1 =   z.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        z2 =   z1 -self.bias
        z3 =   z2 @ self.orth_matrix.weight
        z4 =   z3.permute(0, 3, 1, 2)
        return z4

class NormalizingBlock(Flow):
    def __init__(self,in_channels = 192):
        super().__init__()
        half_channels = in_channels // 2;
        assert half_channels *2 == in_channels

        self.internal_transform = nn.Sequential(
            nn.Conv2d(half_channels,half_channels,groups=half_channels,stride=1, kernel_size = 7, padding = 7//2),
            nn.LeakyReLU(),
            nn.Conv2d(half_channels,half_channels,kernel_size=1,stride=1),
            nn.LeakyReLU()
        )
        self.shuffle = Permute(in_channels)
    def forward(self,z:torch.Tensor):
        z = self.shuffle(z)
        z1,z2 = z.chunk(2,1)
        y1 = z1
        y2 = self.internal_transform(z1)+z2
        return torch.cat([y1, y2], dim=1)

    def inverse(self, z:torch.Tensor):
        z1,z2 = z.chunk(2,1)
        y1 = z1
        y2 = -self.internal_transform(z1)+z2
        
        z = torch.cat([y1, y2], dim=1)
        z = self.shuffle.inverse(z)
        return z

class QNormalizingBlock(Flow):
    def __init__(self,in_channels = 192):
        super().__init__()
        half_channels = in_channels // 2;
        assert half_channels *2 == in_channels

        self.internal_transform = nn.Sequential(
            nn.Conv2d(half_channels,half_channels,groups=half_channels,stride=1, kernel_size = 7, padding = 7//2),
            nn.LeakyReLU(),
            nn.Conv2d(half_channels,half_channels,kernel_size=1,stride=1),
            nn.LeakyReLU()
        )
        self.shuffle = Permute(in_channels,mode="swap")
        self.orth_flow = OrthogonalTransform(half_channels)
    def forward(self,z:torch.Tensor):
        z = self.shuffle(z)
        z1,z2 = z.chunk(2,1)
        y1 = self.orth_flow(z1)
        y2 = self.internal_transform(z1)+z2
        return torch.cat([y1, y2], dim=1)

    def inverse(self, z:torch.Tensor):
        z1,z2 = z.chunk(2,1)
        z1_tilda = self.orth_flow.inverse(z1)
        y1 = z1_tilda
        y2 = -self.internal_transform(z1_tilda)+z2
        
        z = torch.cat([y1, y2], dim=1)
        z = self.shuffle.inverse(z)
        return z


#%%

if __name__ == "__main__":
    net = Composite([QNormalizingBlock(192) for a in range(10)])
    inp = torch.empty(1,192,32,32).random_(10)
    out_1 = net(inp)
    backw = net.inverse(out_1)
    
    print(nn.MSELoss()(inp,backw))

