import torch
import numpy as np

class ImageCollator:
    def __call__(self, examples): # The __call__ method allows the class instance to be called as a function  
        pixel_values = torch.stack([x["pixel_values"] for x in examples]) #stack the tensors of images into a single tensor
        labels = torch.tensor([x["labels"] for x in examples], dtype=torch.long) # convert the labels to a tensor

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
        
