from datasets import load_dataset, DatasetDict

def load_imagefolder(path):
    ds = load_dataset(
        "imagefolder",
        data_dir=path,
        split={
            "train": "train",
            'validation': 'validation',
            "test": "test"
        }
    )

    def replace_image_with_path(example):
        example["path"] = example["image"].filename
        del example["image"]
        return example

    ds = ds.map(replace_image_with_path)

    return DatasetDict(ds)
