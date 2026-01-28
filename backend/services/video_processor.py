"""
Video Processing Service for uploaded video analysis.
Processes video files frame-by-frame using OpenCV.
"""

import cv2
import numpy as np
from typing import List, Dict, Generator, Optional, Any
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Data class for frame processing results."""
    frame_number: int
    timestamp_ms: float
    detections: List[Dict]
    plate_texts: List[str]
    match_statuses: List[bool]
    processing_time_ms: float


class VideoProcessor:
    """
    Video processor for analyzing uploaded video files.
    Extracts frames and processes them through detection pipeline.
    """
    
    def __init__(self, frame_skip: int = None):
        """
        Initialize video processor.
        
        Args:
            frame_skip: Number of frames to skip between processing
        """
        self.frame_skip = frame_skip or settings.FRAME_SKIP
        self.detection_service = None
        self.ocr_service = None
    
    def _init_services(self):
        """Lazy initialization of detection and OCR services."""
        if self.detection_service is None:
            from services.detection import get_detection_service
            self.detection_service = get_detection_service()
        
        if self.ocr_service is None:
            from services.ocr import get_ocr_service
            self.ocr_service = get_ocr_service()
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get metadata about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        return info
    
    def extract_frames(
        self, 
        video_path: str, 
        frame_skip: int = None
    ) -> Generator[tuple, None, None]:
        """
        Generator to extract frames from video.
        
        Args:
            video_path: Path to video file
            frame_skip: Frames to skip (overrides default)
            
        Yields:
            Tuple of (frame_number, timestamp_ms, frame_image)
        """
        skip = frame_skip if frame_skip is not None else self.frame_skip
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_number % (skip + 1) == 0:
                    timestamp_ms = (frame_number / fps) * 1000
                    yield frame_number, timestamp_ms, frame
                
                frame_number += 1
                
        finally:
            cap.release()
    
    def process_frame(
        self, 
        frame: np.ndarray,
        stolen_plates: set = None
    ) -> tuple:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: Video frame (BGR format)
            stolen_plates: Set of normalized stolen plate numbers
            
        Returns:
            Tuple of (detections, plate_texts, match_statuses)
        """
        self._init_services()
        
        stolen_plates = stolen_plates or set()
        
        # Run detection pipeline
        detections = self.detection_service.detect_full_pipeline(frame)
        
        plate_texts = []
        match_statuses = []
        
        # Extract text from detected plates
        for detection in detections:
            for plate in detection.get("plates", []):
                if plate.get("cropped_image") is not None:
                    # Enlarge plate image for better OCR
                    plate_img = plate["cropped_image"]
                    h, w = plate_img.shape[:2]
                    
                    # Resize to minimum 300px width for better OCR
                    if w < 300:
                        scale = 300 / w
                        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, 
                                             interpolation=cv2.INTER_CUBIC)
                    
                    # Run OCR with multiple preprocessing methods
                    text, confidence = self.ocr_service.extract_with_multiple_preprocessing(
                        plate_img
                    )
                    plate_texts.append(text)
                    
                    # Check against stolen database
                    is_stolen = text.upper() in stolen_plates if text else False
                    match_statuses.append(is_stolen)
        
        return detections, plate_texts, match_statuses
    
    def process_video(
        self,
        video_path: str,
        stolen_plates: set = None,
        progress_callback: callable = None
    ) -> List[FrameResult]:
        """
        Process entire video file.
        
        Args:
            video_path: Path to video file
            stolen_plates: Set of normalized stolen plate numbers
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of FrameResult objects
        """
        self._init_services()
        
        results = []
        video_info = self.get_video_info(video_path)
        total_frames = video_info["frame_count"]
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {video_info['fps']}")
        
        processed_count = 0
        
        for frame_number, timestamp_ms, frame in self.extract_frames(video_path):
            start_time = time.time()
            
            # Process frame
            detections, plate_texts, match_statuses = self.process_frame(
                frame, stolen_plates
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Only store results if detections found
            if detections:
                result = FrameResult(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    detections=detections,
                    plate_texts=plate_texts,
                    match_statuses=match_statuses,
                    processing_time_ms=processing_time
                )
                results.append(result)
            
            processed_count += 1
            
            if progress_callback:
                progress_callback(frame_number, total_frames)
        
        logger.info(f"Video processing complete. {len(results)} frames with detections.")
        
        return results
    
    def process_video_streaming(
        self,
        video_path: str,
        stolen_plates: set = None
    ) -> Generator[FrameResult, None, None]:
        """
        Process video and yield results as they are ready (streaming).
        
        Args:
            video_path: Path to video file
            stolen_plates: Set of normalized stolen plate numbers
            
        Yields:
            FrameResult for each processed frame with detections
        """
        self._init_services()
        
        for frame_number, timestamp_ms, frame in self.extract_frames(video_path):
            start_time = time.time()
            
            detections, plate_texts, match_statuses = self.process_frame(
                frame, stolen_plates
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if detections:
                yield FrameResult(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    detections=detections,
                    plate_texts=plate_texts,
                    match_statuses=match_statuses,
                    processing_time_ms=processing_time
                )
    
    def process_image(
        self,
        image: np.ndarray,
        stolen_plates: set = None
    ) -> Dict[str, Any]:
        """
        Process a single image.
        
        Args:
            image: Input image (BGR format)
            stolen_plates: Set of normalized stolen plate numbers
            
        Returns:
            Dict with detection results
        """
        self._init_services()
        
        stolen_plates = stolen_plates or set()
        start_time = time.time()
        
        detections, plate_texts, match_statuses = self.process_frame(
            image, stolen_plates
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Draw annotations
        annotated_image = self.detection_service.draw_detections(
            image, detections, plate_texts, match_statuses
        )
        
        return {
            "detections": detections,
            "plate_texts": plate_texts,
            "match_statuses": match_statuses,
            "processing_time_ms": processing_time,
            "annotated_image": annotated_image
        }


# Singleton instance
_video_processor: Optional[VideoProcessor] = None


def get_video_processor() -> VideoProcessor:
    """
    Get or create singleton video processor instance.
    
    Returns:
        VideoProcessor: The video processor instance
    """
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
