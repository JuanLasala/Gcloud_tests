from typing import Callable, Dict, Tuple
import torch.nn.functional as F


def build_multiband_transforms(
    target_channels: int,
    train_augmentations_fn: Callable,
    eval_augmentations_fn: Callable,
    load_multiband_tiff_fn: Callable,
    force_output_size: Tuple[int, int] = None,
) -> Tuple[Callable, Callable]:
    def _force_resize_if_needed(tensor):
        if force_output_size is None:
            return tensor

        target_h, target_w = force_output_size
        if tensor.shape[-2:] == (target_h, target_w):
            return tensor

        return F.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def train_transform_effnet(batch: Dict):
        tensors = [load_multiband_tiff_fn(p) for p in batch["path"]]
        tensors = [train_augmentations_fn(t) for t in tensors]
        tensors = [_force_resize_if_needed(t) for t in tensors]
        return {"pixel_values": tensors, "labels": batch["label"], "path": batch["path"]}

    def eval_transform_effnet(batch: Dict):
        tensors = [load_multiband_tiff_fn(p) for p in batch["path"]]
        tensors = [eval_augmentations_fn(t) for t in tensors]
        tensors = [_force_resize_if_needed(t) for t in tensors]
        return {"pixel_values": tensors, "labels": batch["label"], "path": batch["path"]}

    return train_transform_effnet, eval_transform_effnet


def apply_effnet_transforms(ds: Dict, train_transform: Callable, eval_transform: Callable) -> Dict:
    return {
        "train": ds["train"].with_transform(train_transform),
        'validation': ds['validation'].with_transform(eval_transform),
        "test": ds["test"].with_transform(eval_transform),
    }
