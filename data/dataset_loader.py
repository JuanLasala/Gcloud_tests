from pathlib import Path

from datasets import DatasetDict, Image, load_dataset


def _find_split_dir(root: Path, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if (root / name).is_dir():
            return name
    return None

def load_imagefolder(path):
    root = Path(path)

    train_dir = _find_split_dir(root, ("train", "training"))
    val_dir = _find_split_dir(root, ("validation", "valid", "val", "dev"))
    test_dir = _find_split_dir(root, ("test", "testing", "eval", "evaluation"))

    if train_dir is None:
        raise FileNotFoundError(
            f"No training split folder found under {root}. "
            "Expected one of: train/, training/."
        )

    data_files = {"train": f"{train_dir}/**"}
    if val_dir is not None:
        data_files["validation"] = f"{val_dir}/**"
    if test_dir is not None:
        data_files["test"] = f"{test_dir}/**"

    ds = load_dataset(
        "imagefolder",
        data_dir=path,
        data_files=data_files,
    )

    # Ensure both splits exist for downstream code.
    if "validation" not in ds:
        if "test" in ds:
            ds["validation"] = ds["test"]
        else:
            split = ds["train"].train_test_split(test_size=0.1, seed=42)
            ds["train"] = split["train"]
            ds["validation"] = split["test"]
    if "test" not in ds:
        ds["test"] = ds["validation"]

    # Avoid decoding to PIL and keep references to local files.
    ds = ds.cast_column("image", Image(decode=False))

    # Extract file paths without per-example decoding.
    ds = ds.flatten()
    ds = ds.rename_column("image.path", "path")
    if "image.bytes" in ds["train"].column_names:
        ds = ds.remove_columns("image.bytes")

    return DatasetDict(ds)
