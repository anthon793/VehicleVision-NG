"""
Pydantic schemas for API request/response validation.
Defines data transfer objects for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
from enum import Enum


# ============ Enums ============

class VehicleTypeEnum(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class MatchStatusEnum(str, Enum):
    STOLEN = "stolen"
    NOT_STOLEN = "not_stolen"
    UNKNOWN = "unknown"


# ============ Stolen Vehicle Schemas ============

class StolenVehicleCreate(BaseModel):
    """Schema for creating a stolen vehicle record."""
    plate_number: str = Field(..., min_length=4, max_length=20, description="License plate number")
    vehicle_type: VehicleTypeEnum = Field(..., description="Type of vehicle")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "plate_number": "LAG234ABC",
                "vehicle_type": "car",
                "description": "Blue Toyota Camry, reported stolen on 2024-01-15"
            }
        }


class StolenVehicleUpdate(BaseModel):
    """Schema for updating a stolen vehicle record."""
    plate_number: Optional[str] = Field(None, min_length=4, max_length=20)
    vehicle_type: Optional[VehicleTypeEnum] = None
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[int] = Field(None, ge=0, le=1)


class StolenVehicleResponse(BaseModel):
    """Schema for stolen vehicle response."""
    id: int
    plate_number: str
    vehicle_type: str
    date_reported: datetime
    description: Optional[str]
    is_active: int
    
    class Config:
        from_attributes = True


# ============ Detection Schemas ============

class BoundingBox(BaseModel):
    """Schema for bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int


class ImageUploadRequest(BaseModel):
    """Schema for image upload request with base64 data."""
    image_data: str = Field(..., description="Base64 encoded image data")
    return_annotated: bool = Field(True, description="Whether to return annotated image")
    # Geolocation fields (optional)
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="GPS longitude")
    location_accuracy: Optional[float] = Field(None, ge=0, description="GPS accuracy in meters")
    location_name: Optional[str] = Field(None, max_length=255, description="Location name")


class GeolocationData(BaseModel):
    """Schema for geolocation data."""
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    accuracy: Optional[float] = Field(None, ge=0)
    location_name: Optional[str] = Field(None, max_length=255)


class PlateDetection(BaseModel):
    """Schema for a single plate detection result."""
    plate_text: str
    confidence: float
    ocr_confidence: float
    bbox: BoundingBox
    is_stolen: bool
    vehicle_type: Optional[str] = None
    # Enhanced fields
    track_id: Optional[int] = Field(None, description="Object tracking ID")
    is_validated: bool = Field(False, description="Whether plate format was validated")
    processing_level: Optional[str] = Field(None, description="Pipeline processing level")
    quality_score: Optional[float] = Field(None, description="Frame quality score")


class DetectionResponse(BaseModel):
    """Schema for detection API response."""
    success: bool
    detections: List[PlateDetection]
    processing_time_ms: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    has_stolen: bool = False
    annotated_image_base64: Optional[str] = None
    # Enhanced metadata
    quality_metrics: Optional[dict] = Field(None, description="Frame quality assessment")
    tracking_active: bool = Field(False, description="Whether object tracking is active")
    roi_applied: bool = Field(False, description="Whether ROI restriction was applied")


class FrameDetectionResult(BaseModel):
    """Schema for a single frame detection result."""
    frame_number: int
    timestamp_ms: float
    detections: List[PlateDetection]
    processing_time_ms: float


class VideoProcessingResponse(BaseModel):
    """Schema for video processing response."""
    success: bool
    message: str
    video_info: dict
    total_frames_processed: int
    frames_with_detections: int
    detections: List[FrameDetectionResult]
    stolen_vehicles_found: List[str]
    total_processing_time_ms: float


class VideoUploadResponse(BaseModel):
    """Schema for video upload response."""
    success: bool
    message: str
    filename: str
    video_info: dict


# ============ Detection Log Schemas ============

class DetectionLogResponse(BaseModel):
    """Schema for detection log response."""
    id: int
    detected_plate: str
    vehicle_type: Optional[str]
    match_status: str
    confidence_score: Optional[float]
    ocr_confidence: Optional[float]
    source_type: Optional[str]
    source_filename: Optional[str]
    frame_number: Optional[int]
    processing_time_ms: Optional[float]
    timestamp: datetime
    # Geolocation fields
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_accuracy: Optional[float] = None
    location_name: Optional[str] = None
    # Processing metadata
    track_id: Optional[int] = None
    is_validated: bool = False
    processing_level: Optional[str] = None
    quality_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class DetectionStatsResponse(BaseModel):
    """Schema for detection statistics response."""
    total_detections: int
    stolen_matches: int
    not_stolen: int
    unknown: int


# ============ Health Check ============

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    timestamp: datetime
    services: dict


# ============ Pagination ============

class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
