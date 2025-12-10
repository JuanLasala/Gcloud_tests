import torch
import numpy as np

def inspect_fp(model, processor, dataset, fire_index, no_fire_index, device='cuda'):
    """
    Lista Falsos Positivos reales:
    No_Fire (true) → Fire (pred)
    """

    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    results = []

    for i in range(len(dataset)):
        item = dataset[i]
        img = item["image"].convert("RGB")
        true = int(item["label"])

        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = int(probs.argmax())

        # FP: No_Fire → Fire
        if true == no_fire_index and pred == fire_index:
            results.append({
                "index": i,
                "path": item.get("path", f"no_path_idx_{i}"),
                "true": true,
                "pred": pred,
                "prob_pred": float(probs[pred]),
                "probs": probs.tolist()
            })

    # orden descendente por confianza
    results.sort(key=lambda x: x["prob_pred"], reverse=True)

    return results
