"""
Database Service for stolen vehicle and detection log operations.
Provides CRUD operations for database entities.
"""

from sqlalchemy.orm import Session
from typing import List, Optional, Set
from datetime import datetime
import logging

from models import StolenVehicle, VehicleType, DetectionLog, MatchStatus

logger = logging.getLogger(__name__)


class StolenVehicleService:
    """Service for managing stolen vehicle records."""
    
    @staticmethod
    def create(
        db: Session,
        plate_number: str,
        vehicle_type: str,
        description: Optional[str] = None
    ) -> StolenVehicle:
        """
        Register a new stolen vehicle.
        
        Args:
            db: Database session
            plate_number: License plate number
            vehicle_type: Type of vehicle
            description: Optional description
            
        Returns:
            StolenVehicle: Created record
        """
        # Normalize plate number
        normalized_plate = plate_number.upper().replace(" ", "").replace("-", "")
        
        stolen_vehicle = StolenVehicle(
            plate_number=normalized_plate,
            vehicle_type=VehicleType(vehicle_type),
            description=description,
            is_active=1
        )
        
        db.add(stolen_vehicle)
        db.commit()
        db.refresh(stolen_vehicle)
        
        logger.info(f"Registered stolen vehicle: {normalized_plate}")
        return stolen_vehicle
    
    @staticmethod
    def get_by_id(db: Session, vehicle_id: int) -> Optional[StolenVehicle]:
        """Get stolen vehicle by ID."""
        return db.query(StolenVehicle).filter(StolenVehicle.id == vehicle_id).first()
    
    @staticmethod
    def get_by_plate(db: Session, plate_number: str) -> Optional[StolenVehicle]:
        """Get stolen vehicle by plate number."""
        normalized = plate_number.upper().replace(" ", "").replace("-", "")
        return db.query(StolenVehicle).filter(
            StolenVehicle.plate_number == normalized
        ).first()
    
    @staticmethod
    def get_all(db: Session, active_only: bool = True) -> List[StolenVehicle]:
        """Get all stolen vehicles."""
        query = db.query(StolenVehicle)
        if active_only:
            query = query.filter(StolenVehicle.is_active == 1)
        return query.order_by(StolenVehicle.date_reported.desc()).all()
    
    @staticmethod
    def get_all_plates(db: Session) -> Set[str]:
        """Get set of all active stolen plate numbers for quick lookup."""
        vehicles = db.query(StolenVehicle.plate_number).filter(
            StolenVehicle.is_active == 1
        ).all()
        return {v.plate_number for v in vehicles}
    
    @staticmethod
    def update(
        db: Session,
        vehicle_id: int,
        **kwargs
    ) -> Optional[StolenVehicle]:
        """Update a stolen vehicle record."""
        vehicle = db.query(StolenVehicle).filter(
            StolenVehicle.id == vehicle_id
        ).first()
        
        if not vehicle:
            return None
        
        for key, value in kwargs.items():
            if hasattr(vehicle, key):
                if key == "plate_number":
                    value = value.upper().replace(" ", "").replace("-", "")
                elif key == "vehicle_type":
                    value = VehicleType(value)
                setattr(vehicle, key, value)
        
        db.commit()
        db.refresh(vehicle)
        return vehicle
    
    @staticmethod
    def delete(db: Session, vehicle_id: int) -> bool:
        """Delete a stolen vehicle record."""
        vehicle = db.query(StolenVehicle).filter(
            StolenVehicle.id == vehicle_id
        ).first()
        
        if not vehicle:
            return False
        
        db.delete(vehicle)
        db.commit()
        return True
    
    @staticmethod
    def mark_resolved(db: Session, vehicle_id: int) -> Optional[StolenVehicle]:
        """Mark a stolen vehicle as resolved (found)."""
        return StolenVehicleService.update(db, vehicle_id, is_active=0)
    
    @staticmethod
    def check_plate(db: Session, plate_number: str) -> tuple:
        """
        Check if a plate number matches a stolen vehicle.
        
        Returns:
            Tuple of (is_stolen, vehicle_record)
        """
        normalized = plate_number.upper().replace(" ", "").replace("-", "")
        vehicle = db.query(StolenVehicle).filter(
            StolenVehicle.plate_number == normalized,
            StolenVehicle.is_active == 1
        ).first()
        
        return (vehicle is not None, vehicle)


class DetectionLogService:
    """Service for managing detection log records."""
    
    @staticmethod
    def create(
        db: Session,
        detected_plate: str,
        vehicle_type: Optional[str] = None,
        match_status: str = "unknown",
        confidence_score: Optional[float] = None,
        ocr_confidence: Optional[float] = None,
        source_type: Optional[str] = None,
        vehicle_color: Optional[str] = None,
        source_filename: Optional[str] = None,
        frame_number: Optional[int] = None,
        processing_time_ms: Optional[float] = None,
        # Geolocation fields (new)
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_accuracy: Optional[float] = None,
        location_name: Optional[str] = None,
        # Processing metadata (new)
        track_id: Optional[int] = None,
        is_validated: bool = False,
        processing_level: Optional[str] = None,
        quality_score: Optional[float] = None
    ) -> DetectionLog:
        """
        Create a new detection log entry.
        
        Args:
            db: Database session
            detected_plate: Detected plate text
            vehicle_type: Type of vehicle detected
            match_status: stolen, not_stolen, or unknown
            confidence_score: Detection confidence
            ocr_confidence: OCR confidence
            source_type: camera, video, or image
            vehicle_color: Detected vehicle color (e.g., "white", "black")
            source_filename: Source file name if applicable
            frame_number: Frame number if from video
            processing_time_ms: Processing time
            latitude: GPS latitude
            longitude: GPS longitude
            location_accuracy: GPS accuracy in meters
            location_name: Human-readable location name
            track_id: Object tracking ID
            is_validated: Whether plate format was validated
            processing_level: Pipeline processing level
            quality_score: Frame quality score
            
        Returns:
            DetectionLog: Created record
        """
        log = DetectionLog(
            detected_plate=detected_plate.upper().replace(" ", "").replace("-", ""),
            vehicle_type=vehicle_type,
            match_status=MatchStatus(match_status),
            confidence_score=confidence_score,
            ocr_confidence=ocr_confidence,
            source_type=source_type,
            vehicle_color=vehicle_color,
            source_filename=source_filename,
            frame_number=frame_number,
            processing_time_ms=processing_time_ms,
            # Geolocation
            latitude=latitude,
            longitude=longitude,
            location_accuracy=location_accuracy,
            location_name=location_name,
            # Processing metadata
            track_id=track_id,
            is_validated=is_validated,
            processing_level=processing_level,
            quality_score=quality_score
        )
        
        db.add(log)
        db.commit()
        db.refresh(log)
        
        return log
    
    @staticmethod
    def get_all(
        db: Session,
        limit: int = 100,
        offset: int = 0,
        stolen_only: bool = False
    ) -> List[DetectionLog]:
        """Get detection logs with pagination."""
        query = db.query(DetectionLog)
        
        if stolen_only:
            query = query.filter(DetectionLog.match_status == MatchStatus.STOLEN)
        
        return query.order_by(
            DetectionLog.timestamp.desc()
        ).offset(offset).limit(limit).all()
    
    @staticmethod
    def get_by_date_range(
        db: Session,
        start_date: datetime,
        end_date: datetime
    ) -> List[DetectionLog]:
        """Get detection logs within a date range."""
        return db.query(DetectionLog).filter(
            DetectionLog.timestamp >= start_date,
            DetectionLog.timestamp <= end_date
        ).order_by(DetectionLog.timestamp.desc()).all()
    
    @staticmethod
    def get_statistics(db: Session) -> dict:
        """Get detection statistics."""
        total = db.query(DetectionLog).count()
        stolen = db.query(DetectionLog).filter(
            DetectionLog.match_status == MatchStatus.STOLEN
        ).count()
        not_stolen = db.query(DetectionLog).filter(
            DetectionLog.match_status == MatchStatus.NOT_STOLEN
        ).count()
        
        return {
            "total_detections": total,
            "stolen_matches": stolen,
            "not_stolen": not_stolen,
            "unknown": total - stolen - not_stolen
        }
    
    @staticmethod
    def get_recent_stolen(db: Session, limit: int = 10) -> List[DetectionLog]:
        """Get recent stolen vehicle detections."""
        return db.query(DetectionLog).filter(
            DetectionLog.match_status == MatchStatus.STOLEN
        ).order_by(DetectionLog.timestamp.desc()).limit(limit).all()
