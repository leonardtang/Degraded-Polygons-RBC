import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MLPMixer(nn.Module):
    def __init__(self, *, num_classes=1000, image_size=(224, 224), channels=3,
                 patch_size=16, dim=512, depth=18, expansion_factor=4,
                 expansion_factor_token=0.5, dropout=0.):
        super().__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.layers = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def load_from(self, weights):
        with torch.no_grad():
            for name, module in self.layers.named_modules():
                if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
                    weight_name = f"{name}/kernel" if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) else f"{name}/scale"
                    bias_name = f"{name}/bias"
                    module.weight.copy_(np2th(weights[weight_name], conv=isinstance(module, nn.Conv1d)))
                    module.bias.copy_(np2th(weights[bias_name]))