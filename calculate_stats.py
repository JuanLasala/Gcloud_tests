import numpy as np
import rasterio
from google.cloud import storage
from io import BytesIO
from tqdm import tqdm
import os

def calculate_dataset_stats(bucket_name, prefix="dataset/train/", output_file="normalization_stats_2.txt"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Filter for .tif files
    tiff_blobs = [b for b in blobs if b.name.endswith(('.tif', '.tiff'))]
    
    num_channels = 6
    pixel_count = 0
    sum_val = np.zeros(num_channels)
    sum_sq_val = np.zeros(num_channels)

    print(f"Processing {len(tiff_blobs)} images from gs://{bucket_name}/{prefix}")

    for blob in tqdm(tiff_blobs):
        try:
            content = blob.download_as_bytes()
            with rasterio.open(BytesIO(content)) as src:
                # Read shape: (6, H, W)
                data = src.read().astype(np.float32)

                # Example: apply to IR channels (adjust indices!)
                ir_indices = [3, 4, 5]  # or whichever are IR

                for c in ir_indices:
                    data[c] = np.log1p(data[c])
                                
                for c in range(num_channels):
                    sum_val[c] += np.sum(data[c])
                    sum_sq_val[c] += np.sum(data[c]**2)
                
                pixel_count += data.shape[1] * data.shape[2]
        except Exception as e:
            print(f"Skipping {blob.name} due to error: {e}")

    # Calculate final Mean and Std
    mean = sum_val / pixel_count
    std = np.sqrt((sum_sq_val / pixel_count) - (mean**2))

    # --- Save to .txt file ---
    # We stack them so the file has two rows: first row is means, second is stds
    stats_matrix = np.vstack([mean, std])
    np.savetxt(output_file, stats_matrix, delimiter=',', 
               header="Row 1: Means (6 bands), Row 2: Stds (6 bands)")
    
    print(f"\nStats saved successfully to {output_file}")
    return mean, std

# Execute
BUCKET = "bucket_six_bands"
calculate_dataset_stats(BUCKET)