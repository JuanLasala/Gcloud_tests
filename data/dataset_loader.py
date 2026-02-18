from datasets import load_dataset, DatasetDict

def load_imagefolder(path):
    ds = load_dataset(
        "imagefolder",
        data_dir=path,
        split={
            "train": "train",
            'validation': 'validation',
            "test": "test"
        },
        decode=False  # Keep file paths instead of decoding images with PIL
    )

    ### Rename "image" column to "path" for compatibility with load_multiband_tiff
    ds = ds.rename_column("image", "path")

    return DatasetDict(ds)
