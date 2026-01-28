"""
Detection Log model for storing detection history.
Captures all detected license plates for analysis and evaluation.

Enhanced with geolocation support for tracking detection locations.
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Enum, Boolean
from sqlalchemy.sql import func
from database.connection import Base
import enum


class MatchStatus(str, enum.Enum):
    """Enumeration for detection match status."""
    STOLEN = "stolen"
    NOT_STOLEN = "not_stolen"
    UNKNOWN = "unknown"


class DetectionLog(Base):
    """
    Detection Log model for storing all detection events.
    
    Attributes:
        id: Primary key, auto-incremented
        detected_plate: The extracted license plate text
        vehicle_type: Detected vehicle type
        match_status: Whether the plate matched a stolen vehicle
        confidence_score: Detection confidence (0.0 - 1.0)
        source_type: Source of detection (camera, video, image)
        frame_number: Frame number if from video
        timestamp: Time of detection
        
    Geolocation fields (new):
        latitude: GPS latitude of detection location
        longitude: GPS longitude of detection location
        location_accuracy: GPS accuracy in meters
        location_name: Human-readable location name
        
    Processing metadata (new):
        track_id: Object tracking ID for multi-frame consistency
        is_validated: Whether plate format was validated
        processing_level: Pipeline processing level used
        quality_score: Frame quality assessment score
    """
    __tablename__ = "detection_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    detected_plate = Column(String(20), index=True, nullable=False)
    vehicle_type = Column(String(20), nullable=True)
    match_status = Column(Enum(MatchStatus), default=MatchStatus.UNKNOWN)
    confidence_score = Column(Float, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    source_type = Column(String(20), nullable=True)  # camera, video, image
    vehicle_color = Column(String(30), nullable=True)  # Detected vehicle color (e.g., "white", "black")
    source_filename = Column(String(255), nullable=True)
    frame_number = Column(Integer, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Geolocation fields
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)  # GPS accuracy in meters
    location_name = Column(String(255), nullable=True)  # e.g., "Lagos, Nigeria"
    
    # Enhanced processing metadata
    track_id = Column(Integer, nullable=True)  # For object tracking
    is_validated = Column(Boolean, default=False)  # Plate format validation
    processing_level = Column(String(20), nullable=True)  # minimal/standard/enhanced/intensive
    quality_score = Column(Float, nullable=True)  # Frame quality 0-1
    
    def __repr__(self):
        return f"<DetectionLog(id={self.id}, plate='{self.detected_plate}', status='{self.match_status}')>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "detected_plate": self.detected_plate,
            "vehicle_type": self.vehicle_type,
            "match_status": self.match_status.value if self.match_status else None,
            "confidence_score": self.confidence_score,
            "ocr_confidence": self.ocr_confidence,
            "source_type": self.source_type,
            "vehicle_color": self.vehicle_color,
            "source_filename": self.source_filename,
            "frame_number": self.frame_number,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_accuracy": self.location_accuracy,
            "location_name": self.location_name,
            "track_id": self.track_id,
            "is_validated": self.is_validated,
            "processing_level": self.processing_level,
            "quality_score": self.quality_score,
        }
