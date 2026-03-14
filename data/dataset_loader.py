from pathlib import Path

from datasets import DatasetDict, Image, load_dataset


def load_imagefolder(path):
    root = Path(path)

    required_splits = ("train", "validation", "test")
    missing = [split for split in required_splits if not (root / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing required split folders under {root}: {missing}. "
            "Expected: train/, validation/, test/."
        )

    ds = load_dataset(
        "imagefolder",
        data_dir=str(root),
        data_files={
            "train": "train/**",
            "validation": "validation/**",
            "test": "test/**",
        },
    )

    ds = ds.cast_column("image", Image(decode=False))

    def add_path(example):
        image_field = example["image"]
        if isinstance(image_field, dict):
            image_path = image_field.get("path")
        else:
            image_path = str(image_field)
        return {"path": image_path}

    ds = ds.map(add_path)

    if "label" not in ds["train"].column_names and "labels" in ds["train"].column_names:
        for split in ("train", "validation", "test"):
            ds[split] = ds[split].rename_column("labels", "label")

    if "label" not in ds["train"].column_names:
        raise KeyError(
            "No 'label' column found in dataset. "
            f"Available columns: {ds['train'].column_names}"
        )

    return DatasetDict(ds)
