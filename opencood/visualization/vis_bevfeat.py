import builtins
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def bev_feature_to_map(bev_feature, normalize=False, reduce="mean"):
    if torch.is_tensor(bev_feature):
        bev_feature = bev_feature.detach().cpu().numpy()

    if bev_feature.ndim == 4:
        bev_feature = bev_feature[0]

    if normalize:
        denom = bev_feature.max() - bev_feature.min()
        if denom > 0:
            bev_feature = (bev_feature - bev_feature.min()) / denom

    if bev_feature.ndim == 3:
        if reduce == "mean":
            bev_map = bev_feature.mean(axis=0)
        elif reduce == "sum":
            bev_map = bev_feature.sum(axis=0)
        else:
            raise ValueError("reduce must be 'mean' or 'sum'")
    elif bev_feature.ndim == 2:
        bev_map = bev_feature
    else:
        raise ValueError("bev_feature must have 2, 3, or 4 dimensions")

    return bev_map


def vis_bev(
    bev_feature,
    type,
    min=None,
    max=None,
    normalize=False,
    save_path=None,
    cmap="viridis",
    upsample=None,
    target_shape=None,
    interpolation="bicubic",
    dpi=500,
):
    bev_map = bev_feature_to_map(bev_feature, normalize=normalize, reduce="mean")

    if target_shape is not None:
        target_h, target_w = target_shape
        if target_h > 0 and target_w > 0:
            bev_map = cv2.resize(
                bev_map,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_CUBIC,
            )
    elif upsample is not None and upsample > 1:
        bev_map = cv2.resize(
            bev_map,
            (int(bev_map.shape[1] * upsample), int(bev_map.shape[0] * upsample)),
            interpolation=cv2.INTER_CUBIC,
        )

    vmin = bev_map.min() if min is None else min
    vmax = bev_map.max() if max is None else max

    if target_shape is not None and dpi > 0:
        fig_w = builtins.max(1.0, bev_map.shape[1] / float(dpi))
        fig_h = builtins.max(1.0, bev_map.shape[0] / float(dpi))
        plt.figure(figsize=(fig_w, fig_h))
    else:
        plt.figure(figsize=(10, 5))

    plt.imshow(
        bev_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation=interpolation,
    )
    plt.axis("off")
    plt.colorbar()

    if save_path is None:
        save_path = f"{type}.png"
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    return bev_map


def visualize_feature_maps(
    feature_map: torch.Tensor,
    mode: str = "mean",
    channel: int = 0,
    cmap: str = "viridis",
    figsize=(10, 4),
    save_path: str = None,
):
    """
    Visualize two feature maps (ego and CAV) with a shared color scale.
    """
    assert feature_map.ndim == 4 and feature_map.shape[0] == 2, "Expected shape [2, C, H, W]"

    if mode == "mean":
        ego_feat = feature_map[0].mean(dim=0).cpu().numpy()
        cav_feat = feature_map[1].mean(dim=0).cpu().numpy()
        title_suffix = " (Mean over Channels)"
    elif mode == "single":
        ego_feat = feature_map[0, channel].cpu().numpy()
        cav_feat = feature_map[1, channel].cpu().numpy()
        title_suffix = f" (Channel {channel})"
    else:
        raise ValueError("mode must be 'mean' or 'single'")

    vmin = min(ego_feat.min(), cav_feat.min())
    vmax = max(ego_feat.max(), cav_feat.max())

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title("Ego Feature" + title_suffix)
    plt.imshow(ego_feat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("CAV Feature" + title_suffix)
    plt.imshow(cav_feat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
