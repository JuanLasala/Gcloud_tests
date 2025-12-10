import torch

class EfficientNetCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # batch es una lista de dicts {"image": PIL, "label": int}
        images = [example["image"] for example in batch]
        labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)

        inputs = self.processor(images=images, return_tensors="pt")

        inputs["labels"] = labels
        return inputs
