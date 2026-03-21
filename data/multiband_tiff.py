import numpy as np
import torch
import tifffile
import logging
import rasterio
from contextlib import contextmanager



class _SuppressGDALNoDataWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "parsing GDAL_NODATA tag raised ValueError" not in msg


@contextmanager
def _quiet_tifffile_gdal_nodata_warning():
    logger = logging.getLogger("tifffile")
    warning_filter = _SuppressGDALNoDataWarning()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)

def _ensure_channel_last(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]

    # If first dimension looks like channels (small) and last looks spatial (large)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
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
        # Some GeoTIFFs store GDAL_NODATA as "0.0", which may trigger a non-fatal tifffile warning.
        with _quiet_tifffile_gdal_nodata_warning():
            arr = tifffile.imread(path)
    except ValueError as err:
        if "requires the 'imagecodecs' package" in str(err):
            raise RuntimeError(
                "TIFF decoding failed because imagecodecs is missing. "
                "Install it with: pip install imagecodecs"
            ) from err
        raise

    arr = _ensure_channel_last(arr)
    return normalize_for_efficientnet(arr)


def make_rgb_preview(tensor: torch.Tensor, bands=(0, 1, 2)) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError("Expected tensor shape (C, H, W).")

    max_band = tensor.shape[0] - 1
    bands = [min(b, max_band) for b in bands]
    rgb = tensor[bands].detach().cpu().numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (np.transpose(rgb, (1, 2, 0)) * 255.0).astype(np.uint8)
    return rgb


# Load once at the start of your script
stats = np.loadtxt("normalization_stats_2.txt", delimiter=',')
GLOBAL_MEAN = stats[0]
GLOBAL_STD = stats[1]

IR_INDICES = [3, 4, 5]  # MUST match stats script


def normalize_for_efficientnet(arr: np.ndarray) -> torch.Tensor:
    """
    Input: (H, W, 6)
    Output: (6, H, W)
    """
    arr = arr.astype(np.float32)

    # 🔥 Apply SAME log transform as stats computation
    for c in IR_INDICES:
        arr[..., c] = np.log1p(arr[..., c])

    # 📊 Global normalization
    arr = (arr - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)

    # 🔄 Convert to PyTorch format
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    return tensor

# --- 2. Usage in a Dataset Class ---
""" class WildfireDataset(Dataset):
    def __init__(self, tiff_paths, labels, transform=None):
        self.tiff_paths = tiff_paths  # List of paths (local or GCS fuse)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.tiff_paths)

    def __getitem__(self, idx):
        path = self.tiff_paths[idx]
        label = self.labels[idx]

        # --- This is where raw_tiff_data comes from ---
        with rasterio.open(path) as src:
            # src.read() returns a NumPy array of shape (Channels, Height, Width)
            # For 6 bands, this is (6, H, W)
            raw_tiff_data = src.read().astype('float32')

        # Convert to PyTorch Tensor
        img_tensor = torch.from_numpy(raw_tiff_data)

        # Apply your custom_normalize (and any other transforms)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(label, dtype=torch.long) """