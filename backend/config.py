"""
Configuration module for the Stolen Vehicle Detection System.
Loads environment variables and provides centralized configuration.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    """Application settings loaded from environment variables."""
    
    # Database - Default to SQLite for easy local development
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{BASE_DIR / 'stolen_vehicle.db'}"
    )
    
    # JWT Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Model Paths
    VEHICLE_MODEL_PATH: str = os.getenv("VEHICLE_MODEL_PATH", str(BASE_DIR / "models" / "yolov8n.pt"))
    PLATE_MODEL_PATH: str = os.getenv("PLATE_MODEL_PATH", str(BASE_DIR / "models" / "plate_detector.pt"))
    
    # Roboflow API (for enhanced plate detection)
    ROBOFLOW_API_KEY: str = os.getenv("ROBOFLOW_API_KEY", "")
    # Optional: Roboflow Workflow configuration
    ROBOFLOW_WORKSPACE: str = os.getenv("ROBOFLOW_WORKSPACE", "")
    ROBOFLOW_WORKFLOW: str = os.getenv("ROBOFLOW_WORKFLOW", "")
    ROBOFLOW_USE_WORKFLOW: bool = os.getenv("ROBOFLOW_USE_WORKFLOW", "true").lower() in ("1", "true", "yes")
    
    # Detection Settings
    DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))
    # FRAME_SKIP: Process every Nth frame (higher = faster but may miss plates)
    # Default 5 = ~6 frames per second for 30fps video (balanced)
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "5"))
    
    # TURBO MODE: Faster video processing (less preprocessing)
    # Only affects video, not single image detection
    TURBO_MODE: bool = os.getenv("TURBO_MODE", "true").lower() in ("1", "true", "yes")
    
    # Image size for processing (1024 is good balance of speed and accuracy)
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
    
    # Upload Settings
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_VIDEO_EXTENSIONS: set = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    # Vehicle Classes (COCO dataset classes for vehicles)
    VEHICLE_CLASSES: dict = {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
    
    # Email Alert Configuration
    EMAIL_ALERTS_ENABLED: bool = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() in ("1", "true", "yes")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")  # Verified sender for Brevo
    ALERT_EMAIL_RECIPIENTS: str = os.getenv("ALERT_EMAIL_RECIPIENTS", "")  # Comma-separated

settings = Settings()

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
