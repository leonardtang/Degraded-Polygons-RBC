import copy
from os.path import join as pjoin
import numpy as np
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from torchvision import models
from vit_pytorch import SimpleViT
from vit_pytorch.efficient import ViT
from linformer import Linformer
import ml_collections

pair = lambda x: x if isinstance(x, tuple) else (x, x)

def get_mixer_b16_config():
    """Returns Mixer-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_dim = 768
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    return config

def get_model(args, classes):
    if args.model == "vgg16":
        if not args.evaluate:
            if args.pretrained_imagenet:
                model = models.vgg16(pretrained=True)
            else:
                model = models.vgg16()
            features = []
            for feat in list(model.features):
                features.append(feat)
                if isinstance(feat, nn.Conv2d):
                    features.append(nn.Dropout(p=0.55, inplace=True))
            
            model.features = nn.Sequential(*features)
            model.fc = torch.nn.Linear(512, len(classes))
        else:
            model = models.vgg16()
            model.fc = torch.nn.Linear(512, len(classes))

    elif args.model == "resnet18":
        if not args.evaluate:
            if args.pretrained_imagenet:
                model = models.resnet18(pretrained=True)
            elif args.pretrained_path:
                model = models.resnet18()
                if args.pretrained_path.endswith("FractalDB-10000_res18.pth"):
                    model.fc = torch.nn.Linear(512, 10000)        
                elif args.pretrained_path.endswith("FractalDB-1000_res18.pth"):
                    model.fc = torch.nn.Linear(512, 1000)
                try:
                    model.load_state_dict(torch.load(args.pretrained_path))
                except:
                    model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
            else:
                model = models.resnet18()
            model.fc = torch.nn.Linear(512, len(classes))
        else:
            model = models.resnet18()
            model.fc = torch.nn.Linear(512, len(classes))

    elif args.model == "resnet50":
        if not args.evaluate:
            if args.pretrained_imagenet:
                model = models.resnet50(pretrained=True)
            elif args.pretrained_path:
                raise Exception('Not implemented')
                # model = models.resnet50()
                # if args.pretrained_path.endswith("FractalDB-10000_res18.pth"):
                #     model.fc = torch.nn.Linear(2048, 10000)        
                # elif args.pretrained_path.endswith("FractalDB-1000_res18.pth"):
                #     model.fc = torch.nn.Linear(2048, 1000)
                # try:
                #     model.load_state_dict(torch.load(args.pretrained_path))
                # except:
                #     model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
            else:
                model = models.resnet50()
            model.fc = torch.nn.Linear(2048, len(classes))
        else:
            model = models.resnet50()
            model.fc = torch.nn.Linear(2048, len(classes))

    elif args.model == "mlpmixer":  # work in progress
        config = get_mixer_b16_config()
        if not args.evaluate:
            if args.pretrained_path:  # assumes imagenet1k
                model = MLPMixer(config, num_classes=1000)
                model.load_from(np.load(args.pretrained_path))
                # re-initialize final layer
                # model.layers[-1] = nn.Linear(model.layers[-1].in_features, len(classes))
                model.head = nn.Linear(config.hidden_dim, len(classes), bias=True)
            else:
                model = MLPMixer(config, num_classes=len(classes))
        else:
            model = MLPMixer(config, num_classes=len(classes))

    elif args.model == "vit":
        # TODO(ltang): think carefully about which ViT to base off of
        if args.pretrained_imagenet:
            model = models.vit_b_16(weights='IMAGENET1K_V1')
            # re-initialize final layer
            model.heads.head = nn.Linear(model.heads.head.in_features, len(classes), bias=True)
            
        else:
            model = models.vit_b_16(num_classes=len(classes))
        
        # efficient_transformer = Linformer(
        #     dim=128,
        #     # seq_len=49+1,  # 7x7 patches + 1 cls-token
        #     seq_len=14*14+1,  # 14x14 patches + 1 cls-token
        #     depth=12,
        #     heads=8,
        #     k=64
        # )

        # model = ViT(
        #     dim=128,
        #     image_size=224,
        #     patch_size=16,
        #     num_classes=len(classes),
        #     transformer=efficient_transformer,
        #     channels=3,
        # ).to(device)

        # model = SimpleViT(
        #     image_size = 224,
        #     patch_size = 32,
        #     num_classes = len(classes),
        #     dim = 1024,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048
        # )

    else:
        raise Exception(f"{args.model} is not a supported model")
    
    return model


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super(MlpBlock, self).__init__()
        self.fc0 = nn.Linear(hidden_dim, ff_dim, bias=True)
        self.fc1 = nn.Linear(ff_dim, hidden_dim, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.fc0(x)
        x = self.act_fn(x)
        x = self.fc1(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(config.n_patches, config.tokens_mlp_dim)
        self.channel_mlp_block = MlpBlock(config.hidden_dim, config.channels_mlp_dim)
        self.pre_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.post_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"MixerBlock_{n_block}"
        with torch.no_grad():
            self.token_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "kernel")]).t())
            self.token_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "kernel")]).t())
            self.token_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "bias")]).t())
            self.token_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "bias")]).t())

            self.channel_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "kernel")]).t())
            self.channel_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "kernel")]).t())
            self.channel_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "bias")]).t())
            self.channel_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "bias")]).t())

            self.pre_norm.weight.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "scale")]))
            self.pre_norm.bias.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "bias")]))
            self.post_norm.weight.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "scale")]))
            self.post_norm.bias.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "bias")]))


class MLPMixer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, patch_size=16, zero_head=False):
        super(MLPMixer, self).__init__()
        self.zero_head = zero_head
        self.num_classes = num_classes
        patch_size = _pair(patch_size)
        n_patches = (img_size // patch_size[0]) * (img_size // patch_size[1])
        config.n_patches = n_patches

        self.stem = nn.Conv2d(in_channels=3,
                              out_channels=config.hidden_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.head = nn.Linear(config.hidden_dim, num_classes, bias=True)
        self.pre_head_ln = nn.LayerNorm(config.hidden_dim, eps=1e-6)


        self.layer = nn.ModuleList()
        for _ in range(config.num_blocks):
            layer = MixerBlock(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, labels=None):
        x = self.stem(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        for block in self.layer:
            x = block(x)
        x = self.pre_head_ln(x)
        x = torch.mean(x, dim=1)
        logits = self.head(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
            self.stem.weight.copy_(np2th(weights["stem/kernel"], conv=True))
            self.stem.bias.copy_(np2th(weights["stem/bias"]))
            self.pre_head_ln.weight.copy_(np2th(weights["pre_head_layer_norm/scale"]))
            self.pre_head_ln.bias.copy_(np2th(weights["pre_head_layer_norm/bias"]))

            for bname, block in self.layer.named_children():
                block.load_from(weights, n_block=bname)

TOK_FC_0 = "token_mixing/Dense_0"
TOK_FC_1 = "token_mixing/Dense_1"
CHA_FC_0 = "channel_mixing/Dense_0"
CHA_FC_1 = "channel_mixing/Dense_1"
PRE_NORM = "LayerNorm_0"
POST_NORM = "LayerNorm_1"


# class PreNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         return self.fn(self.norm(x)) + x


# def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
#     inner_dim = int(dim * expansion_factor)
#     return nn.Sequential(
#         dense(dim, inner_dim),
#         nn.GELU(),
#         nn.Dropout(dropout),
#         dense(inner_dim, dim),
#         nn.Dropout(dropout)
#     )


# class MLPMixer(nn.Module):
#     def __init__(self, *, num_classes=1000, image_size=(224, 224), channels=3,
#                  patch_size=16, dim=512, depth=18, expansion_factor=4,
#                  expansion_factor_token=0.5, dropout=0.):
#         super().__init__()
#         image_h, image_w = pair(image_size)
#         assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
#         num_patches = (image_h // patch_size) * (image_w // patch_size)
#         chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

#         self.layers = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
#             nn.Linear((patch_size ** 2) * channels, dim),
#             *[nn.Sequential(
#                 PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
#                 PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
#             ) for _ in range(depth)],
#             nn.LayerNorm(dim),
#             Reduce('b n c -> b c', 'mean'),
#             nn.Linear(dim, num_classes)
#         )
    
#     def forward(self, x):
#         return self.layers(x)
    
#     def load_from(self, weights):
#         with torch.no_grad():
#             for name, module in self.layers.named_modules():
#                 if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
#                     weight_name = f"{name}/kernel" if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) else f"{name}/scale"
#                     bias_name = f"{name}/bias"
#                     print(name, module, weight_name, bias_name)
#                     print("KEYS", weights.keys())
#                     module.weight.copy_(np2th(weights[weight_name], conv=isinstance(module, nn.Conv1d)))
#                     module.bias.copy_(np2th(weights[bias_name]))