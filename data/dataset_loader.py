from datasets import load_dataset, DatasetDict

def load_imagefolder(path):
    ds_full = load_dataset("imagefolder", data_dir=path)
    # Add a new column "path" to the dataset.
    # Return ONLY the new key so datasets preserves all original feature types
    # (ClassLabel, Image, etc.) without re-inference.
    def add_path(example):
        img = example["image"]
        path = getattr(img, "filename", None) or ""
        return {"path": path}
    ds_full = ds_full.map(add_path)
    # Return the modified dataset as a DatasetDict
    return DatasetDict({
        "train": ds_full["train"],
        "validation": ds_full["validation"],
        "test": ds_full["test"]
    })
