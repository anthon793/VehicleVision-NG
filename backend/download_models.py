"""
Script to download YOLOv8 models from Roboflow for license plate detection.
Run this script once to download the models.
"""

import os
from pathlib import Path

# Create models directory
models_dir = Path(__file__).parent.parent / "models"
models_dir.mkdir(exist_ok=True)

print(f"Models directory: {models_dir}")

# Option 1: Using Roboflow inference API (recommended for quick setup)
print("\n=== Setting up Roboflow Inference ===")
print("The license plate detection will use Roboflow's hosted inference API.")
print("This provides the best accuracy with the pre-trained model.")

# Create a config file for Roboflow settings
config_content = """
# Roboflow Configuration for License Plate Detection
# 
# License Plate Detection Model:
# - Project: muhammad-najib-sulaiman/lpd-pfzpe
# - Model Version: 5
# - Metrics: mAP@50 99.5%, Precision 99.5%, Recall 99.5%
# - Trained with YOLOv9
#
# To use the Roboflow API:
# 1. Sign up at https://roboflow.com
# 2. Get your API key from Settings -> API Key
# 3. Set ROBOFLOW_API_KEY environment variable
#
# Alternatively, download the model weights:
# 1. Go to https://universe.roboflow.com/muhammad-najib-sulaiman/lpd-pfzpe/model/5
# 2. Click "Download" -> Select "YOLOv8" format
# 3. Place the .pt file in this models directory as "plate_detector.pt"

ROBOFLOW_API_KEY=your_api_key_here
LPD_PROJECT=muhammad-najib-sulaiman/lpd-pfzpe
LPD_VERSION=5
"""

config_path = models_dir / "roboflow_config.txt"
with open(config_path, "w") as f:
    f.write(config_content)

print(f"\nConfig file created: {config_path}")
print("\n=== Instructions ===")
print("1. Get a free Roboflow API key from: https://app.roboflow.com/settings/api")
print("2. Add to your .env file: ROBOFLOW_API_KEY=your_key_here")
print("\nOr download the model directly:")
print("1. Visit: https://universe.roboflow.com/muhammad-najib-sulaiman/lpd-pfzpe/model/5")
print("2. Click 'Download this Model' -> 'YOLOv8' format")
print(f"3. Save as: {models_dir / 'plate_detector.pt'}")
print("\nThe system will automatically use the model once configured.")
