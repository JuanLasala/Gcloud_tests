import torch
import numpy as np

class ImageCollator:
    def __call__(self, examples):
        # examples es la lista de lotes (batch) donde cada elemento es un diccionario que contiene los tensores de imagen ya procesados por with_transform.
        pixel_values = torch.stack([x["pixel_values"] for x in examples]) # Apilar a lo largo de una nueva dimensión (batch_size, 3, 224, 224)
        labels = torch.tensor([x["labels"] for x in examples], dtype=torch.long) # Convertir a tensor de tipo long

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
        
