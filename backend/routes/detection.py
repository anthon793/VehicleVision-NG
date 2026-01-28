"""
Detection Routes.
Handles image/frame detection and video processing endpoints.
"""

import cv2
import numpy as np
import base64
import time
import os
import uuid
import aiofiles
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import logging

from database import get_db, SessionLocal
from models import User
from auth import get_current_user
from services import get_video_processor, get_detection_service, get_ocr_service
from services.database_service import StolenVehicleService, DetectionLogService
from services.color_detection import get_color_service
from services.email_service import get_email_service
# Brevo transactional email (preferred over SMTP)
from utils.brevo_email import send_missing_vehicle_email
from routes.schemas import (
    DetectionResponse, PlateDetection, BoundingBox,
    VideoProcessingResponse, FrameDetectionResult, VideoUploadResponse,
    ImageUploadRequest
)
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detection", tags=["Detection"])


def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    # Remove data URL prefix if present
    if "," in image_data:
        image_data = image_data.split(",")[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy array image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


@router.post("/frame", response_model=DetectionResponse)
async def detect_frame(
    image_data: str,
    return_annotated: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process a single frame/image for vehicle and license plate detection.
    
    - Accepts base64 encoded image
    - Returns detected plates with bounding boxes
    - Checks plates against stolen vehicle database
    - Optionally returns annotated image
    """
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(image_data)
        height, width = image.shape[:2]
        
        # Get services
        processor = get_video_processor()
        
        # Get stolen plates for comparison
        stolen_plates = StolenVehicleService.get_all_plates(db)
        
        # Process frame
        result = processor.process_image(image, stolen_plates)
        
        # Build response
        detections = []
        has_stolen = False
        
        for i, detection in enumerate(result["detections"]):
            for j, plate in enumerate(detection.get("plates", [])):
                plate_idx = sum(len(d.get("plates", [])) for d in result["detections"][:i]) + j
                
                if plate_idx < len(result["plate_texts"]):
                    plate_text = result["plate_texts"][plate_idx]
                    is_stolen = result["match_statuses"][plate_idx] if plate_idx < len(result["match_statuses"]) else False
                    
                    if is_stolen:
                        has_stolen = True
                    
                    bbox = plate.get("bbox", (0, 0, 0, 0))
                    
                    plate_detection = PlateDetection(
                        plate_text=plate_text,
                        confidence=detection["vehicle_confidence"],
                        ocr_confidence=0.0,  # OCR confidence not tracked separately here
                        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
                        is_stolen=is_stolen,
                        vehicle_type=detection["vehicle_type"]
                    )
                    detections.append(plate_detection)
                    
                    # Log detection
                    DetectionLogService.create(
                        db=db,
                        detected_plate=plate_text,
                        vehicle_type=detection["vehicle_type"],
                        match_status="stolen" if is_stolen else "not_stolen",
                        confidence_score=detection["vehicle_confidence"],
                        source_type="camera",
                        processing_time_ms=result["processing_time_ms"]
                    )
                    
                    # Send email alert if stolen vehicle detected
                    if is_stolen:
                        # Use SMTP email service
                        email_service = get_email_service()
                        alert_image = encode_image_base64(result["annotated_image"]) if result.get("annotated_image") is not None else None
                        email_service.send_alert_async(
                            plate_number=plate_text,
                            vehicle_type=detection["vehicle_type"],
                            vehicle_color=detection.get("vehicle_color"),
                            confidence=detection["vehicle_confidence"],
                            source_type="camera",
                            image_base64=alert_image
                        )
                        logger.info(f"Email alert queued for stolen vehicle: {plate_text}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Encode annotated image if requested
        annotated_base64 = None
        if return_annotated:
            annotated_base64 = encode_image_base64(result["annotated_image"])
        
        return DetectionResponse(
            success=True,
            detections=detections,
            processing_time_ms=processing_time,
            frame_width=width,
            frame_height=height,
            has_stolen=has_stolen,
            annotated_image_base64=annotated_base64
        )
        
    except Exception as e:
        logger.error(f"Frame detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.post("/frame-upload")
async def detect_frame_upload(
    request: ImageUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process a single image for vehicle and license plate detection.
    Accepts image data in request body for larger images.
    
    - Accepts base64 encoded image in JSON body
    - Returns detected plates with bounding boxes
    - Checks plates against stolen vehicle database
    - Optionally returns annotated image
    """
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_data)
        height, width = image.shape[:2]
        
        # Get services
        processor = get_video_processor()
        
        # Get stolen plates for comparison
        stolen_plates = StolenVehicleService.get_all_plates(db)
        
        # Process frame
        result = processor.process_image(image, stolen_plates)
        
        # Build response
        plates = []
        has_stolen = False
        
        for i, detection in enumerate(result["detections"]):
            for j, plate in enumerate(detection.get("plates", [])):
                plate_idx = sum(len(d.get("plates", [])) for d in result["detections"][:i]) + j
                
                if plate_idx < len(result["plate_texts"]):
                    plate_text = result["plate_texts"][plate_idx]
                    is_stolen = result["match_statuses"][plate_idx] if plate_idx < len(result["match_statuses"]) else False
                    
                    if is_stolen:
                        has_stolen = True
                    
                    bbox = plate.get("bbox", (0, 0, 0, 0))
                    
                    plates.append({
                        "plate_text": plate_text,
                        "confidence": detection["vehicle_confidence"],
                        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                        "is_stolen": is_stolen,
                        "vehicle_type": detection["vehicle_type"]
                    })
                    
                    # Log detection
                    DetectionLogService.create(
                        db=db,
                        detected_plate=plate_text,
                        vehicle_type=detection["vehicle_type"],
                        match_status="stolen" if is_stolen else "not_stolen",
                        confidence_score=detection["vehicle_confidence"],
                        source_type="image_upload",
                        processing_time_ms=result["processing_time_ms"]
                    )
                    
                    # Send email alert if stolen vehicle detected
                    if is_stolen:
                        # Use SMTP email service
                        email_service = get_email_service()
                        alert_image = encode_image_base64(result["annotated_image"]) if result.get("annotated_image") is not None else None
                        email_service.send_alert_async(
                            plate_number=plate_text,
                            vehicle_type=detection["vehicle_type"],
                            vehicle_color=detection.get("vehicle_color"),
                            confidence=detection["vehicle_confidence"],
                            source_type="image_upload",
                            image_base64=alert_image
                        )
                        logger.info(f"Email alert queued for stolen vehicle: {plate_text}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Encode annotated image if requested
        annotated_base64 = None
        if request.return_annotated:
            annotated_base64 = encode_image_base64(result["annotated_image"])
        
        return {
            "success": True,
            "plates": plates,
            "processing_time_ms": processing_time,
            "frame_width": width,
            "frame_height": height,
            "has_stolen": has_stolen,
            "annotated_image": annotated_base64
        }
        
    except Exception as e:
        logger.error(f"Frame upload detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a video file for processing.
    
    - Accepts MP4, AVI, MOV, MKV, WebM formats
    - Maximum file size: 100MB
    - Returns video metadata
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}"
        )
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    try:
        # Save file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            await out_file.write(content)
        
        # Get video info
        processor = get_video_processor()
        video_info = processor.get_video_info(file_path)
        video_info["filename"] = unique_filename
        video_info["original_filename"] = file.filename
        
        return VideoUploadResponse(
            success=True,
            message="Video uploaded successfully",
            filename=unique_filename,
            video_info=video_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Video upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/process-video/{filename}", response_model=VideoProcessingResponse)
async def process_uploaded_video(
    filename: str,
    frame_skip: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process an uploaded video file.
    
    - Analyzes video frame by frame
    - Returns all detections with timestamps
    - Identifies stolen vehicles
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found"
        )
    
    start_time = time.time()
    
    try:
        processor = get_video_processor()
        if frame_skip:
            processor.frame_skip = frame_skip
        
        # Get stolen plates
        stolen_plates = StolenVehicleService.get_all_plates(db)
        
        # Get video info
        video_info = processor.get_video_info(file_path)
        
        # Process video
        results = processor.process_video(file_path, stolen_plates)
        
        # Build response
        frame_results = []
        stolen_found = set()
        
        for result in results:
            frame_detections = []
            
            for i, detection in enumerate(result.detections):
                for j, plate in enumerate(detection.get("plates", [])):
                    plate_idx = sum(len(d.get("plates", [])) for d in result.detections[:i]) + j
                    
                    if plate_idx < len(result.plate_texts):
                        plate_text = result.plate_texts[plate_idx]
                        is_stolen = result.match_statuses[plate_idx] if plate_idx < len(result.match_statuses) else False
                        
                        if is_stolen and plate_text:
                            stolen_found.add(plate_text)
                        
                        bbox = plate.get("bbox", (0, 0, 0, 0))
                        
                        frame_detections.append(PlateDetection(
                            plate_text=plate_text,
                            confidence=detection.get("vehicle_confidence", 0),
                            ocr_confidence=0.0,
                            bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
                            is_stolen=is_stolen,
                            vehicle_type=detection.get("vehicle_type")
                        ))
                        
                        # Log detection
                        DetectionLogService.create(
                            db=db,
                            detected_plate=plate_text,
                            vehicle_type=detection.get("vehicle_type"),
                            match_status="stolen" if is_stolen else "not_stolen",
                            confidence_score=detection.get("vehicle_confidence"),
                            source_type="video",
                            source_filename=filename,
                            frame_number=result.frame_number,
                            processing_time_ms=result.processing_time_ms
                        )
            
            if frame_detections:
                frame_results.append(FrameDetectionResult(
                    frame_number=result.frame_number,
                    timestamp_ms=result.timestamp_ms,
                    detections=frame_detections,
                    processing_time_ms=result.processing_time_ms
                ))
        
        total_time = (time.time() - start_time) * 1000
        
        return VideoProcessingResponse(
            success=True,
            message="Video processed successfully",
            video_info=video_info,
            total_frames_processed=len(results),
            frames_with_detections=len(frame_results),
            detections=frame_results,
            stolen_vehicles_found=list(stolen_found),
            total_processing_time_ms=total_time
        )
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@router.get("/stream-video/{filename}")
async def stream_video_detections(
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Stream video processing results as Server-Sent Events.
    
    - Real-time detection updates
    - Useful for long videos
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found"
        )
    
    async def event_generator():
        try:
            processor = get_video_processor()
            stolen_plates = StolenVehicleService.get_all_plates(db)
            
            for result in processor.process_video_streaming(file_path, stolen_plates):
                event_data = {
                    "frame_number": result.frame_number,
                    "timestamp_ms": result.timestamp_ms,
                    "plate_texts": result.plate_texts,
                    "match_statuses": result.match_statuses,
                    "processing_time_ms": result.processing_time_ms
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
            
            yield f"data: {json.dumps({'complete': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.delete("/video/{filename}")
async def delete_video(
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete an uploaded video file.
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found"
        )
    
    try:
        os.remove(file_path)
        return {"success": True, "message": "Video deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete video: {str(e)}"
        )


# ============ ENHANCED DETECTION ENDPOINTS ============

@router.post("/frame-enhanced")
async def detect_frame_enhanced(
    request: ImageUploadRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Enhanced frame detection with all advanced features:
    - Geolocation support
    - Object tracking
    - ROI restriction
    - Quality-based adaptive processing
    - Nigerian plate format validation
    - Confidence-driven pipeline
    
    Academic Note: This endpoint demonstrates the enhanced pipeline
    for BSc research evaluation.
    """
    from services.tracking import get_tracker
    from services.roi_processor import get_roi_processor
    from services.image_quality import get_quality_processor
    from services.plate_validator import process_ocr_result
    
    start_time = time.time()
    
    try:
        # Decode image
        image = decode_base64_image(request.image_data)
        height, width = image.shape[:2]
        
        # Get services
        processor = get_video_processor()
        tracker = get_tracker()
        roi_processor = get_roi_processor()
        quality_processor = get_quality_processor()
        
        # Assess frame quality
        quality_metrics = quality_processor.assess_quality(image)
        quality_dict = {
            "blur_score": quality_metrics.blur_score,
            "brightness": quality_metrics.brightness,
            "contrast": quality_metrics.contrast,
            "is_blurry": quality_metrics.is_blurry,
            "overall_quality": quality_metrics.overall_quality
        }
        
        # Apply quality enhancement if needed
        process_image = image
        if quality_metrics.overall_quality in ["medium", "poor"]:
            process_image = quality_processor.enhance_image(image, quality_metrics)
        
        # Apply ROI restriction (optional - for performance)
        roi_applied = False
        roi_coords = None
        # Uncomment below to enable ROI:
        # process_image, roi_coords = roi_processor.crop_roi(process_image)
        # roi_applied = True
        
        # Get stolen plates for comparison
        stolen_plates = StolenVehicleService.get_all_plates(db)
        
        # Process frame
        result = processor.process_image(process_image, stolen_plates)
        
        # Build detections for tracking
        raw_detections = []
        for detection in result["detections"]:
            for plate in detection.get("plates", []):
                raw_detections.append({
                    "bbox": plate.get("bbox", (0, 0, 0, 0)),
                    "confidence": detection["vehicle_confidence"],
                    "class_name": detection["vehicle_type"],
                    "plate_text": plate.get("text")
                })
        
        # Update tracker
        tracked_objects = tracker.update(raw_detections)
        tracking_active = len(tracked_objects) > 0
        
        # Build response
        plates = []
        has_stolen = False
        color_service = get_color_service()
        email_service = get_email_service()
        
        for i, detection in enumerate(result["detections"]):
            # Detect vehicle color from the vehicle's cropped image
            vehicle_color = "unknown"
            vehicle_color_confidence = 0.0
            vehicle_cropped = detection.get("cropped_image")
            if vehicle_cropped is not None:
                vehicle_color, vehicle_color_confidence = color_service.detect_vehicle_color(vehicle_cropped)
            
            for j, plate in enumerate(detection.get("plates", [])):
                plate_idx = sum(len(d.get("plates", [])) for d in result["detections"][:i]) + j
                
                if plate_idx < len(result["plate_texts"]):
                    plate_text = result["plate_texts"][plate_idx]
                    is_stolen = result["match_statuses"][plate_idx] if plate_idx < len(result["match_statuses"]) else False
                    
                    if is_stolen:
                        has_stolen = True
                        # Send email alert for stolen vehicle (async, non-blocking)
                        email_service.send_alert_async(
                            plate_number=plate_text,
                            vehicle_type=detection["vehicle_type"],
                            vehicle_color=vehicle_color,
                            confidence=detection["vehicle_confidence"],
                            location=request.location_name,
                            image_base64=request.image_data if request.return_annotated else None,
                            source_type="camera_enhanced"
                        )
                    
                    # Validate plate format
                    validation = process_ocr_result(plate_text)
                    validated_text = validation.corrected_text
                    is_validated = validation.is_valid
                    
                    # Adjust confidence based on validation
                    adjusted_confidence = detection["vehicle_confidence"]
                    if is_validated:
                        adjusted_confidence = min(1.0, adjusted_confidence + validation.confidence_boost)
                    
                    # Get track ID if available
                    track_id = None
                    for tracked in tracked_objects:
                        if tracked.plate_text == plate_text:
                            track_id = tracked.track_id
                            break
                    
                    bbox = plate.get("bbox", (0, 0, 0, 0))
                    
                    plates.append({
                        "plate_text": validated_text,
                        "confidence": adjusted_confidence,
                        "ocr_confidence": plate.get("ocr_confidence", 0.0),
                        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                        "is_stolen": is_stolen,
                        "vehicle_type": detection["vehicle_type"],
                        "vehicle_color": vehicle_color,
                        "vehicle_color_confidence": vehicle_color_confidence,
                        "track_id": track_id,
                        "is_validated": is_validated,
                        "processing_level": "enhanced" if quality_metrics.overall_quality != "good" else "standard",
                        "quality_score": quality_metrics.blur_score / 500  # Normalize to 0-1
                    })
                    
                    # Log detection with geolocation and enhanced metadata
                    DetectionLogService.create(
                        db=db,
                        detected_plate=validated_text,
                        vehicle_type=detection["vehicle_type"],
                        match_status="stolen" if is_stolen else "not_stolen",
                        confidence_score=adjusted_confidence,
                        ocr_confidence=plate.get("ocr_confidence"),
                        source_type="camera_enhanced",
                        vehicle_color=vehicle_color,
                        processing_time_ms=result["processing_time_ms"],
                        # Geolocation
                        latitude=request.latitude,
                        longitude=request.longitude,
                        location_accuracy=request.location_accuracy,
                        location_name=request.location_name,
                        # Enhanced metadata
                        track_id=track_id,
                        is_validated=is_validated,
                        processing_level="enhanced" if quality_metrics.overall_quality != "good" else "standard",
                        quality_score=quality_metrics.blur_score / 500
                    )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Encode annotated image if requested
        annotated_base64 = None
        if request.return_annotated:
            annotated_base64 = encode_image_base64(result["annotated_image"])
        
        return {
            "success": True,
            "detections": plates,
            "processing_time_ms": processing_time,
            "frame_width": width,
            "frame_height": height,
            "has_stolen": has_stolen,
            "annotated_image": annotated_base64,
            # Enhanced metadata
            "quality_metrics": quality_dict,
            "tracking_active": tracking_active,
            "roi_applied": roi_applied,
            "geolocation": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "accuracy": request.location_accuracy,
                "name": request.location_name
            } if request.latitude and request.longitude else None
        }
        
    except Exception as e:
        logger.error(f"Enhanced detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced detection failed: {str(e)}"
        )


@router.post("/reset-tracking")
async def reset_object_tracking(
    current_user: User = Depends(get_current_user)
):
    """
    Reset the object tracking state.
    Call this when switching video sources or starting a new session.
    """
    from services.tracking import reset_tracker
    
    try:
        reset_tracker()
        return {"success": True, "message": "Tracking state reset successfully"}
    except Exception as e:
        logger.error(f"Tracking reset error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset tracking: {str(e)}"
        )


@router.get("/quality-check")
async def check_frame_quality(
    image_data: str,
    current_user: User = Depends(get_current_user)
):
    """
    Check frame quality without running full detection.
    Useful for camera preview to advise users on image quality.
    """
    from services.image_quality import get_quality_processor
    
    try:
        image = decode_base64_image(image_data)
        quality_processor = get_quality_processor()
        metrics = quality_processor.assess_quality(image)
        
        return {
            "success": True,
            "quality": {
                "blur_score": metrics.blur_score,
                "brightness": metrics.brightness,
                "contrast": metrics.contrast,
                "is_blurry": metrics.is_blurry,
                "is_dark": metrics.is_dark,
                "is_low_contrast": metrics.is_low_contrast,
                "overall_quality": metrics.overall_quality
            },
            "recommendations": _get_quality_recommendations(metrics)
        }
    except Exception as e:
        logger.error(f"Quality check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality check failed: {str(e)}"
        )


def _get_quality_recommendations(metrics) -> list:
    """Generate recommendations based on quality metrics."""
    recommendations = []
    
    if metrics.is_blurry:
        recommendations.append("Image appears blurry. Hold camera steady or improve focus.")
    
    if metrics.is_dark:
        recommendations.append("Image appears too dark or overexposed. Adjust lighting conditions.")
    
    if metrics.is_low_contrast:
        recommendations.append("Low contrast detected. Ensure clear view of vehicle.")
    
    if metrics.overall_quality == "good":
        recommendations.append("Image quality is good for detection.")
    
    return recommendations


# ============================================================
# WEBSOCKET ENDPOINT FOR REAL-TIME VIDEO PROCESSING
# ============================================================

@router.websocket("/ws/process-video/{filename}")
async def websocket_process_video(websocket: WebSocket, filename: str, token: str = None):
    """
    WebSocket endpoint for real-time video processing updates.
    
    Sends frame-by-frame detection results as the video is processed.
    Token authentication via query parameter: ?token=xxx
    
    Message format sent to client:
    {
        "type": "frame_result" | "progress" | "complete" | "error",
        "data": {...}
    }
    """
    # Validate token before accepting connection
    from auth.utils import decode_token
    
    # Must accept WebSocket first before sending any response
    await websocket.accept()
    
    if not token:
        await websocket.send_json({"type": "error", "message": "Missing authentication token"})
        await websocket.close(code=4001)
        return
    
    try:
        payload = decode_token(token)
        if not payload:
            await websocket.send_json({"type": "error", "message": "Invalid token"})
            await websocket.close(code=4001)
            return
    except Exception as e:
        logger.warning(f"WebSocket auth failed: {e}")
        await websocket.send_json({"type": "error", "message": "Authentication failed"})
        await websocket.close(code=4001)
        return
    
    logger.info(f"WebSocket connected for video processing: {filename}")
    
    db = SessionLocal()
    
    try:
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        if not os.path.exists(file_path):
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Video file not found"}
            })
            await websocket.close()
            return
        
        # Get video info
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Send video info
        await websocket.send_json({
            "type": "video_info",
            "data": {
                "filename": filename,
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": duration,
                "width": width,
                "height": height
            }
        })
        
        # Get processor and stolen plates
        processor = get_video_processor()
        stolen_plates = StolenVehicleService.get_all_plates(db)
        
        frames_processed = 0
        detections_found = 0
        stolen_found = set()
        all_detections = []  # Collect all detections for final summary
        unique_plates = {}  # Track unique plates with their first detection
        start_time = time.time()
        
        # Process video and stream results
        stolen_vehicle_detected = False
        stolen_detection_info = None
        
        for result in processor.process_video_streaming(file_path, stolen_plates):
            frames_processed += 1
            
            # Build frame result
            frame_detections = []
            for i, (plate_text, is_stolen) in enumerate(zip(result.plate_texts, result.match_statuses)):
                if plate_text:
                    detections_found += 1
                    
                    # Get confidence and vehicle info from detection if available
                    confidence = 0.8  # Default
                    vehicle_type = "vehicle"
                    vehicle_color = "unknown"
                    if result.detections and i < len(result.detections):
                        det = result.detections[i]
                        if isinstance(det, dict):
                            vehicle_type = det.get('vehicle_type', 'vehicle')
                            vehicle_color = det.get('vehicle_color', 'unknown')
                            if 'plates' in det and det['plates']:
                                confidence = det['plates'][0].get('confidence', 0.8)
                    
                    detection_data = {
                        "plate_text": plate_text,
                        "is_stolen": is_stolen,
                        "confidence": confidence,
                        "vehicle_type": vehicle_type,
                        "vehicle_color": vehicle_color
                    }
                    frame_detections.append(detection_data)
                    
                    # Track unique plates (first occurrence only)
                    normalized_plate = plate_text.upper().replace(' ', '').replace('-', '')
                    if normalized_plate not in unique_plates:
                        unique_plates[normalized_plate] = {
                            "plate_text": plate_text,
                            "is_stolen": is_stolen,
                            "confidence": confidence,
                            "frame_number": result.frame_number,
                            "timestamp_ms": result.timestamp_ms,
                            "vehicle_type": vehicle_type,
                            "vehicle_color": vehicle_color
                        }
                    
                    # If stolen vehicle detected - prepare to stop
                    if is_stolen:
                        stolen_found.add(plate_text)
                        stolen_vehicle_detected = True
                        stolen_detection_info = {
                            "plate_text": plate_text,
                            "frame_number": result.frame_number,
                            "timestamp_ms": result.timestamp_ms,
                            "timestamp_formatted": f"{int(result.timestamp_ms // 60000)}:{int((result.timestamp_ms % 60000) // 1000):02d}.{int((result.timestamp_ms % 1000) // 10):02d}",
                            "confidence": confidence,
                            "vehicle_type": vehicle_type,
                            "vehicle_color": vehicle_color,
                            "video_position_percent": round((result.frame_number / total_frames * 100) if total_frames > 0 else 0, 1)
                        }
                    
                    # Log to database
                    try:
                        DetectionLogService.create(
                            db=db,
                            detected_plate=plate_text,
                            vehicle_type=vehicle_type,
                            match_status="stolen" if is_stolen else "not_stolen",
                            confidence_score=confidence,
                            source_type="video_realtime",
                            source_filename=filename,
                            frame_number=result.frame_number,
                            processing_time_ms=result.processing_time_ms,
                            vehicle_color=vehicle_color
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log detection: {e}")
            
            # Calculate progress
            progress = (frames_processed / total_frames * 100) if total_frames > 0 else 0
            
            # Collect frame detections for final summary
            if frame_detections:
                all_detections.append({
                    "frame_number": result.frame_number,
                    "timestamp_ms": result.timestamp_ms,
                    "detections": frame_detections
                })
            
            # Send frame result
            await websocket.send_json({
                "type": "frame_result",
                "data": {
                    "frame_number": result.frame_number,
                    "timestamp_ms": result.timestamp_ms,
                    "timestamp_formatted": f"{int(result.timestamp_ms // 60000)}:{int((result.timestamp_ms % 60000) // 1000):02d}",
                    "detections": frame_detections,
                    "processing_time_ms": result.processing_time_ms,
                    "progress_percent": round(progress, 1),
                    "frames_processed": frames_processed,
                    "total_detections": detections_found,
                    "stolen_found": list(stolen_found)
                }
            })
            
            # ★ STOP IMMEDIATELY if stolen vehicle found ★
            if stolen_vehicle_detected:
                logger.warning(f"🚨 STOLEN VEHICLE DETECTED: {stolen_detection_info['plate_text']} at frame {stolen_detection_info['frame_number']}")
                
                # Send stolen vehicle alert
                await websocket.send_json({
                    "type": "stolen_found",
                    "data": {
                        "alert": True,
                        "message": f"STOLEN VEHICLE DETECTED! Processing stopped.",
                        "stolen_vehicle": stolen_detection_info,
                        "frames_processed_before_detection": frames_processed,
                        "total_frames_in_video": total_frames,
                        "processing_stopped": True
                    }
                })
                
                # Break out of the loop - stop processing
                break
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        # Send completion message with all detection details
        total_time = (time.time() - start_time) * 1000
        await websocket.send_json({
            "type": "complete",
            "data": {
                "success": True,
                "stopped_early": stolen_vehicle_detected,
                "stopped_reason": "Stolen vehicle detected" if stolen_vehicle_detected else None,
                "stolen_vehicle_info": stolen_detection_info,
                "total_frames_processed": frames_processed,
                "total_frames_in_video": total_frames,
                "frames_with_detections": len(all_detections),
                "total_detections": detections_found,
                "stolen_vehicles_found": list(stolen_found),
                "total_processing_time_ms": total_time,
                "avg_frame_time_ms": total_time / frames_processed if frames_processed > 0 else 0,
                "detections": all_detections,
                "unique_plates": list(unique_plates.values())
            }
        })
        
        logger.info(f"Video processing complete: {frames_processed} frames, {detections_found} detections")
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket video processing error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass
    finally:
        db.close()
        try:
            await websocket.close()
        except:
            pass

