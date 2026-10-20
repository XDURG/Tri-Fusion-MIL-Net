import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from infer import load_model
from train import BRANCHES, collate_cases, get_device, load_case, move_batch


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-8)


def save_overlay(image, cam, path):
    gray = normalize(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    color = plt.get_cmap("jet")(cam)[..., :3]
    alpha = 0.55 * cam[..., None]
    overlay = rgb * (1 - alpha) + color * alpha
    panel = np.concatenate((rgb, overlay), axis=1)
    Image.fromarray(np.uint8(np.clip(panel, 0, 1) * 255)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", default="weights.pt")
    parser.add_argument("--out", default="heatmaps")
    parser.add_argument("--class-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = get_device(args.device)
    model = load_model(args.weights, device)
    case = load_case(args.input)
    batch, _, _ = collate_cases([(case, float("nan"), Path(args.input).stem)])
    batch = move_batch(batch, device)
    activations = {key: [] for key in BRANCHES}

    def capture(key):
        def hook(module, inputs, output):
            output.retain_grad()
            activations[key].append(output)
        return hook

    handles = [model.planes[key].network.conv_head.register_forward_hook(capture(key)) for key in BRANCHES[:3]]
    handles.append(model.volume.projection.register_forward_hook(capture("patches")))
    try:
        result = model(batch, return_attention=True)
        target = result["logits"].sum() * (1 if args.class_index == 1 else -1)
        target.backward()
    finally:
        for handle in handles:
            handle.remove()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in BRANCHES:
        maps = []
        for activation in activations[key]:
            if activation.grad is None:
                raise RuntimeError(f"No gradient recorded for {key}")
            axes = tuple(range(2, activation.ndim))
            weights = activation.grad.mean(dim=axes, keepdim=True)
            maps.append((activation * weights).sum(dim=1, keepdim=True).relu().detach())
        maps = torch.cat(maps)
        mode = "trilinear" if key == "patches" else "bilinear"
        maps = F.interpolate(maps, size=case[key].shape[2:], mode=mode, align_corners=False)[:, 0].cpu().numpy()
        images = case[key][:, 0].numpy()
        for i, (image, cam) in enumerate(zip(images, maps)):
            cam = normalize(cam)
            if key == "patches":
                for axis, name in enumerate(("axial", "coronal", "sagittal")):
                    index = image.shape[axis] // 2
                    save_overlay(np.take(image, index, axis=axis), np.take(cam, index, axis=axis), out / f"patch_{i + 1:03d}_{name}.png")
            else:
                save_overlay(image, cam, out / f"{key}_{i + 1:03d}.png")
    for key in ("attention_2d_to_3d", "attention_3d_to_2d"):
        matrix = result[key][0].detach().float().mean(0).cpu().numpy()
        plt.imsave(out / (key + ".png"), matrix, cmap="viridis", vmin=0)
    print(f"carcinoma_probability={result['logits'].detach().sigmoid().item():.6f}")
    print(out)


if __name__ == "__main__":
    main()
