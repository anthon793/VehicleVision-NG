"""
Detection Logs Routes.
Endpoints for viewing detection history and statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from database import get_db
from models import User
from auth import get_current_user, get_admin_user
from services.database_service import DetectionLogService
from routes.schemas import DetectionLogResponse, DetectionStatsResponse

router = APIRouter(prefix="/logs", tags=["Detection Logs"])


@router.get("/", response_model=List[DetectionLogResponse])
async def get_detection_logs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    stolen_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detection logs with pagination.
    
    - Returns most recent detections first
    - Optional filter for stolen vehicles only
    """
    logs = DetectionLogService.get_all(
        db, 
        limit=limit, 
        offset=offset, 
        stolen_only=stolen_only
    )
    
    return [
        DetectionLogResponse(
            id=log.id,
            detected_plate=log.detected_plate,
            vehicle_type=log.vehicle_type,
            match_status=log.match_status.value,
            confidence_score=log.confidence_score,
            ocr_confidence=log.ocr_confidence,
            source_type=log.source_type,
            source_filename=log.source_filename,
            frame_number=log.frame_number,
            processing_time_ms=log.processing_time_ms,
            timestamp=log.timestamp
        )
        for log in logs
    ]


@router.get("/statistics", response_model=DetectionStatsResponse)
async def get_detection_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detection statistics overview.
    
    - Total detections
    - Stolen matches
    - Non-stolen detections
    """
    stats = DetectionLogService.get_statistics(db)
    
    return DetectionStatsResponse(**stats)


@router.get("/recent-stolen", response_model=List[DetectionLogResponse])
async def get_recent_stolen_detections(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent stolen vehicle detections.
    
    - Returns most recent stolen vehicle matches
    - Useful for alerts dashboard
    """
    logs = DetectionLogService.get_recent_stolen(db, limit=limit)
    
    return [
        DetectionLogResponse(
            id=log.id,
            detected_plate=log.detected_plate,
            vehicle_type=log.vehicle_type,
            match_status=log.match_status.value,
            confidence_score=log.confidence_score,
            ocr_confidence=log.ocr_confidence,
            source_type=log.source_type,
            source_filename=log.source_filename,
            frame_number=log.frame_number,
            processing_time_ms=log.processing_time_ms,
            timestamp=log.timestamp
        )
        for log in logs
    ]


@router.get("/by-date", response_model=List[DetectionLogResponse])
async def get_logs_by_date_range(
    start_date: datetime = Query(..., description="Start date (ISO format)"),
    end_date: datetime = Query(..., description="End date (ISO format)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user)
):
    """
    Get detection logs within a date range (Admin only).
    
    - Useful for generating reports
    - Dates in ISO format: YYYY-MM-DDTHH:MM:SS
    """
    if end_date < start_date:
        raise HTTPException(
            status_code=400,
            detail="End date must be after start date"
        )
    
    logs = DetectionLogService.get_by_date_range(db, start_date, end_date)
    
    return [
        DetectionLogResponse(
            id=log.id,
            detected_plate=log.detected_plate,
            vehicle_type=log.vehicle_type,
            match_status=log.match_status.value,
            confidence_score=log.confidence_score,
            ocr_confidence=log.ocr_confidence,
            source_type=log.source_type,
            source_filename=log.source_filename,
            frame_number=log.frame_number,
            processing_time_ms=log.processing_time_ms,
            timestamp=log.timestamp
        )
        for log in logs
    ]


@router.get("/today", response_model=List[DetectionLogResponse])
async def get_todays_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all detection logs from today.
    """
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    logs = DetectionLogService.get_by_date_range(db, today_start, today_end)
    
    return [
        DetectionLogResponse(
            id=log.id,
            detected_plate=log.detected_plate,
            vehicle_type=log.vehicle_type,
            match_status=log.match_status.value,
            confidence_score=log.confidence_score,
            ocr_confidence=log.ocr_confidence,
            source_type=log.source_type,
            source_filename=log.source_filename,
            frame_number=log.frame_number,
            processing_time_ms=log.processing_time_ms,
            timestamp=log.timestamp
        )
        for log in logs
    ]
