import torch

class EfficientNetCollator:
    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, batch):
        labels_key = "labels" if "labels" in batch[0] else "label"
        labels = torch.tensor([example[labels_key] for example in batch], dtype=torch.long)

        if "pixel_values" in batch[0]:
            pixel_values = torch.stack([example["pixel_values"] for example in batch])
            return {"pixel_values": pixel_values, "labels": labels}

        images = [example["image"] for example in batch]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs
