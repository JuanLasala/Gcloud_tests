# config.py
CFG = {
    "model_name": "google/efficientnet-v2-s",
    "data_path": "/home/jlasala/ViT tests",
    "img_size": 380,
    "in_channels": 7,  # 5 actual + 2 placeholders
    "actual_channels": 5,
    "batch_size": 16,
    "epochs": 10,
    "lr": 5e-5,
}