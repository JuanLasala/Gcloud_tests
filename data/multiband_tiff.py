import numpy as np
import torch
import tifffile

def _ensure_channel_last(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]

    # If shape is (C, H, W), convert to (H, W, C)
    if arr.ndim == 3 and arr.shape[0] == 11:
        arr = np.transpose(arr, (1, 2, 0))

    return arr


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    # 2nd and 98th percentile clipping
    p2, p98 = np.percentile(arr, (2, 98))
    arr = np.clip(arr, p2, p98)
    return (arr - p2) / (p98 - p2 + 1e-8)


def load_multiband_tiff(path: str) -> torch.Tensor:
    try:
        arr = tifffile.imread(path)
    except ValueError as err:
        if "requires the 'imagecodecs' package" in str(err):
            raise RuntimeError(
                "TIFF decoding failed because imagecodecs is missing. "
                "Install it with: pip install imagecodecs"
            ) from err
        raise

    arr = _ensure_channel_last(arr)
    arr = _normalize_array(arr)

    # Optional but recommended: reduce RAM pressure
    arr = arr.astype(np.float16)

    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def make_rgb_preview(tensor: torch.Tensor, bands=(0, 1, 2)) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError("Expected tensor shape (C, H, W).")

    max_band = tensor.shape[0] - 1
    bands = [min(b, max_band) for b in bands]
    rgb = tensor[bands].detach().cpu().numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (np.transpose(rgb, (1, 2, 0)) * 255.0).astype(np.uint8)
    return rgb
