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
    arr = arr.astype(np.float32)

    # Normalize each band independently
    for c in range(arr.shape[-1]):
        band = arr[..., c]

        p2, p98 = np.percentile(band, (2, 98))
        band = np.clip(band, p2, p98)

        # Avoid division by zero
        denom = (p98 - p2)
        if denom < 1e-6:
            denom = 1.0

        arr[..., c] = (band - p2) / denom

    return arr


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

    # Not downgrading to float16 because of precision issues
    arr = arr.astype(np.float32)

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
