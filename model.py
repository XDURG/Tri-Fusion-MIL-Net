import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False),
            group_norm(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            group_norm(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
            group_norm(out_channels),
        )

    def forward(self, x):
        return F.gelu(self.layers(x) + self.skip(x))


class ResidualViT3D(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2, 1, bias=False),
            group_norm(32),
            nn.GELU(),
        )
        layers = []
        current = 32
        for i, width in enumerate((32, 64, 128, 256)):
            for j in range(2):
                stride = 2 if i > 0 and j == 0 else 1
                layers.append(ResidualBlock3D(current, width, stride))
                current = width
        self.stages = nn.Sequential(*layers)
        self.projection = nn.Conv3d(256, dim, 1)
        self.position = nn.Linear(3, dim)
        layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 4, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.projection(self.stages(self.stem(x)))
        x = F.adaptive_avg_pool3d(x, (3, 3, 3))
        axis = torch.linspace(-1, 1, 3, device=x.device, dtype=x.dtype)
        coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        x = x.flatten(2).transpose(1, 2) + self.position(coordinates).unsqueeze(0)
        return self.norm(self.transformer(x)).mean(dim=1)


class PlaneEncoder(nn.Module):
    def __init__(self, dim, pretrained):
        super().__init__()
        self.network = timm.create_model("tf_efficientnet_lite0", pretrained=pretrained, in_chans=1, num_classes=0)
        self.projection = nn.Sequential(nn.Linear(self.network.num_features, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return self.projection(self.network(x))


def masked_mean(x, valid):
    return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


class TopKSummary(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.k = k

    def forward(self, features, valid):
        logits = self.score(features).squeeze(-1)
        values, indices = logits.masked_fill(~valid, -torch.inf).topk(min(self.k, features.shape[1]), dim=1)
        selected = features.gather(1, indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        selected_valid = torch.isfinite(values)
        weights = values.masked_fill(~selected_valid, 0).sigmoid() * selected_valid
        selected = selected * weights.unsqueeze(-1)
        selected = F.pad(selected, (0, 0, 0, self.k - selected.shape[1]))
        return selected.flatten(1), logits, indices


class TriFusionMIL(nn.Module):
    def __init__(self, pretrained=False, chunk_size=8):
        super().__init__()
        self.dim = 256
        self.chunk_size = chunk_size
        self.planes = nn.ModuleDict({name: PlaneEncoder(256, pretrained) for name in ("axial", "coronal", "sagittal")})
        self.volume = ResidualViT3D()
        self.cross_2d = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.cross_3d = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.fc_2d = nn.Linear(256, 256)
        self.fc_3d = nn.Linear(256, 256)
        self.gate = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())
        self.top_2d = TopKSummary(256, 5)
        self.top_3d = TopKSummary(256, 5)
        self.classifier = nn.Sequential(nn.Linear(2816, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def encode(self, network, data, valid):
        batch, count = data.shape[:2]
        indices = valid.flatten().nonzero(as_tuple=False).flatten()
        selected = data.flatten(0, 1).index_select(0, indices)
        encoded = []
        for x in selected.split(self.chunk_size):
            if self.training and network is self.volume:
                encoded.append(checkpoint(network, x, use_reentrant=False))
            else:
                encoded.append(network(x))
        encoded = torch.cat(encoded)
        padded = encoded.new_zeros(batch * count, self.dim).index_copy(0, indices, encoded)
        return padded.reshape(batch, count, self.dim)

    def forward(self, batch, return_attention=False):
        features = []
        masks = []
        for plane in ("axial", "coronal", "sagittal"):
            valid = batch[plane + "_valid"]
            features.append(self.encode(self.planes[plane], batch[plane], valid))
            masks.append(valid)
        f = torch.cat(features, dim=1)
        mask_f = torch.cat(masks, dim=1)
        mask_g = batch["patches_valid"]
        g = self.encode(self.volume, batch["patches"], mask_g)
        cross_f, attention_fg = self.cross_2d(
            f, g, g, key_padding_mask=~mask_g, need_weights=return_attention, average_attn_weights=False
        )
        cross_g, attention_gf = self.cross_3d(
            g, f, f, key_padding_mask=~mask_f, need_weights=return_attention, average_attn_weights=False
        )
        pooled_f = masked_mean(cross_f, mask_f)
        pooled_g = masked_mean(cross_g, mask_g)
        alpha = self.gate(torch.cat((pooled_f, pooled_g), dim=-1))
        fused = alpha * self.fc_2d(pooled_f) + (1 - alpha) * self.fc_3d(pooled_g)
        top_f, scores_f, indices_f = self.top_2d(f, mask_f)
        top_g, scores_g, indices_g = self.top_3d(g, mask_g)
        logits = self.classifier(torch.cat((fused, top_f, top_g), dim=-1)).squeeze(-1)
        output = {
            "logits": logits,
            "gate": alpha,
            "instance_2d_logits": scores_f,
            "instance_3d_logits": scores_g,
            "top_2d_indices": indices_f,
            "top_3d_indices": indices_g,
        }
        if return_attention:
            output["attention_2d_to_3d"] = attention_fg
            output["attention_3d_to_2d"] = attention_gf
        return output
