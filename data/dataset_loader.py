from datasets import load_dataset, DatasetDict

def load_imagefolder(path):
    ds_full = load_dataset("imagefolder", data_dir=path)

    def add_path(example):
        example["path"] = example["image"].filename
        return example

    ds_full = ds_full.map(add_path)

    return DatasetDict({
        "train": ds_full["train"],
        "val": ds_full["validation"],
        "test": ds_full["test"]
    })
