from datasets import load_dataset, DatasetDict

def load_imagefolder(path):
    ds_full = load_dataset("imagefolder", data_dir=path)
    # Add a new column "path" to the dataset
    def add_path(example):
        example["path"] = example["image"].filename
        return example
        # Apply the function to add the "path" column to the dataset
    ds_full = ds_full.map(add_path)
    # Return the modified dataset as a DatasetDict
    return DatasetDict({
        "train": ds_full["train"],
        "val": ds_full["validation"],
        "test": ds_full["test"]
    })
