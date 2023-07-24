import torch
import torch.nn as nn
from compressai.models import ScaleHyperprior,get_scale_table
from compressai.models.utils import conv,deconv,update_registered_buffers
from compressai.layers import GDN
from compressai.entropy_models import GaussianConditional
from base import Composite,Flow
from coupled import Permute,QNormalizingBlock
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


class CustomScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=192, **kwargs):
        
        super().__init__(N=N,M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.nf = Composite([QNormalizingBlock(M) for _ in range(10)])
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        u = self.g_a(x)

        y = self.nf(u) #Normalize the feature tensor

        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)

        u_hat = self.nf.inverse(y_hat) #Invert normalization
        x_hat = self.g_s(u_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table() #0.11 to a particular value.
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

model = CustomScaleHyperprior
if __name__=="__main__":
    print("Num params :")
    sample = model()
    num_params= sum([a.numel() for a in sample.parameters() if a.requires_grad])
    print("  {:.3f}M".format(num_params/1e6))