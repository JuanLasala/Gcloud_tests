from datasets import DatasetDict, Image, load_dataset

def load_imagefolder(path):
    ds = load_dataset(
        "imagefolder",
        data_dir=path,
        data_files={
            "train": "train/**",
            "validation": "validation/**",
            "test": "test/**",
        },
    )

    # Avoid decoding to PIL and keep references to local files.
    ds = ds.cast_column("image", Image(decode=False))

    # Extract file paths without per-example decoding.
    ds = ds.flatten()
    ds = ds.rename_column("image.path", "path")
    if "image.bytes" in ds["train"].column_names:
        ds = ds.remove_columns("image.bytes")

    return DatasetDict(ds)
