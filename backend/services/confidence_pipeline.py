"""
Confidence-Driven Detection Pipeline.
Orchestrates detection flow based on confidence levels and quality assessment.

Academic Justification (Chapter 4):
- Adaptive processing saves computation on clear images
- Multiple passes improve accuracy on challenging frames
- Smart resource allocation for real-time performance
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from .tracking import get_tracker, TrackedObject
from .roi_processor import get_roi_processor, ROIConfig
from .image_quality import get_quality_processor, QualityMetrics
from .plate_validator import process_ocr_result, PlateValidationResult

logger = logging.getLogger(__name__)


class ProcessingLevel(Enum):
    """Processing intensity levels."""
    MINIMAL = "minimal"     # Fast pass, cached results
    STANDARD = "standard"   # Normal detection
    ENHANCED = "enhanced"   # Quality enhancement + detection
    INTENSIVE = "intensive" # Full pipeline with re-detection


@dataclass
class PipelineConfig:
    """Configuration for confidence-driven pipeline."""
    # Confidence thresholds
    high_confidence_threshold: float = 0.8    # Skip re-processing
    medium_confidence_threshold: float = 0.5  # Standard processing
    low_confidence_threshold: float = 0.3     # Enhanced processing
    
    # Feature toggles
    enable_tracking: bool = True
    enable_roi: bool = True
    enable_quality_enhancement: bool = True
    enable_plate_validation: bool = True
    enable_smart_zoom: bool = True
    
    # Performance settings
    max_retries: int = 2
    skip_ocr_for_tracked: bool = True  # Use cached OCR for tracked objects
    min_plate_area: int = 500          # Minimum plate area in pixels
    
    # Smart zoom settings
    zoom_factor: float = 2.0
    zoom_on_low_confidence: bool = True


@dataclass
class DetectionCandidate:
    """A detection candidate with metadata."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    plate_text: Optional[str] = None
    ocr_confidence: float = 0.0
    track_id: Optional[int] = None
    quality_metrics: Optional[QualityMetrics] = None
    validation_result: Optional[PlateValidationResult] = None
    processing_level: ProcessingLevel = ProcessingLevel.STANDARD
    is_cached: bool = False


@dataclass
class PipelineResult:
    """Result from the confidence-driven pipeline."""
    detections: List[DetectionCandidate]
    frame_number: int
    processing_time_ms: float
    processing_level: ProcessingLevel
    quality_metrics: Optional[QualityMetrics] = None
    roi_applied: bool = False
    tracking_active: bool = False


class ConfidenceDrivenPipeline:
    """
    Orchestrates detection with confidence-based decision making.
    Integrates tracking, ROI, quality enhancement, and validation.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detection_service: Any = None  # Injected detection service
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            detection_service: The main detection service to use
        """
        self.config = config or PipelineConfig()
        self.detection_service = detection_service
        
        # Initialize sub-components
        self.tracker = get_tracker() if self.config.enable_tracking else None
        self.roi_processor = get_roi_processor() if self.config.enable_roi else None
        self.quality_processor = get_quality_processor() if self.config.enable_quality_enhancement else None
        
        self.frame_count = 0
        self._last_detections: List[DetectionCandidate] = []
        
        logger.info("Confidence-Driven Pipeline initialized")
    
    def determine_processing_level(
        self,
        frame: np.ndarray,
        previous_confidence: Optional[float] = None
    ) -> ProcessingLevel:
        """
        Determine appropriate processing level based on context.
        
        Args:
            frame: Input frame
            previous_confidence: Confidence from previous frame detection
            
        Returns:
            ProcessingLevel to use
        """
        # Check quality
        if self.quality_processor:
            quality = self.quality_processor.assess_quality(frame)
            
            if quality.overall_quality == "poor":
                return ProcessingLevel.INTENSIVE
            elif quality.overall_quality == "medium":
                return ProcessingLevel.ENHANCED
        
        # Check previous confidence
        if previous_confidence is not None:
            if previous_confidence >= self.config.high_confidence_threshold:
                return ProcessingLevel.MINIMAL
            elif previous_confidence >= self.config.medium_confidence_threshold:
                return ProcessingLevel.STANDARD
            elif previous_confidence >= self.config.low_confidence_threshold:
                return ProcessingLevel.ENHANCED
        
        return ProcessingLevel.STANDARD
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        force_level: Optional[ProcessingLevel] = None
    ) -> PipelineResult:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR)
            frame_number: Frame index
            force_level: Force specific processing level
            
        Returns:
            PipelineResult with detections and metadata
        """
        start_time = time.time()
        self.frame_count = frame_number
        
        # Determine processing level
        prev_conf = self._get_previous_avg_confidence()
        level = force_level or self.determine_processing_level(frame, prev_conf)
        
        # Assess frame quality
        quality_metrics = None
        if self.quality_processor:
            quality_metrics = self.quality_processor.assess_quality(frame)
        
        # Apply ROI if enabled
        roi_applied = False
        roi_coords = None
        process_frame = frame
        
        if self.roi_processor and self.config.enable_roi:
            if level in [ProcessingLevel.STANDARD, ProcessingLevel.MINIMAL]:
                process_frame, roi_coords = self.roi_processor.crop_roi(frame)
                roi_applied = True
        
        # Apply quality enhancement if needed
        if level in [ProcessingLevel.ENHANCED, ProcessingLevel.INTENSIVE]:
            if self.quality_processor and quality_metrics:
                process_frame = self.quality_processor.enhance_image(
                    process_frame, quality_metrics
                )
        
        # Run detection
        raw_detections = self._run_detection(process_frame, level)
        
        # Adjust coordinates if ROI was applied
        if roi_applied and roi_coords:
            raw_detections = self.roi_processor.adjust_detections_to_original(
                raw_detections, roi_coords
            )
        
        # Update tracking
        tracking_active = False
        if self.tracker and self.config.enable_tracking:
            tracked = self.tracker.update(raw_detections)
            tracking_active = True
            
            # Convert tracked objects to detection candidates
            candidates = self._tracked_to_candidates(tracked, frame)
        else:
            candidates = self._raw_to_candidates(raw_detections)
        
        # Process OCR and validation for each candidate
        final_detections = []
        for candidate in candidates:
            processed = self._process_candidate(candidate, frame, level)
            final_detections.append(processed)
        
        # Smart zoom re-detection for low-confidence plates
        if self.config.enable_smart_zoom and self.config.zoom_on_low_confidence:
            final_detections = self._apply_smart_zoom(final_detections, frame, level)
        
        # Store for next frame reference
        self._last_detections = final_detections
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            detections=final_detections,
            frame_number=frame_number,
            processing_time_ms=processing_time,
            processing_level=level,
            quality_metrics=quality_metrics,
            roi_applied=roi_applied,
            tracking_active=tracking_active
        )
    
    def _run_detection(
        self,
        frame: np.ndarray,
        level: ProcessingLevel
    ) -> List[Dict]:
        """
        Run detection at specified level.
        
        Note: This method should be overridden or the detection_service
        should be injected to actually perform detection.
        """
        if self.detection_service is None:
            logger.warning("No detection service configured")
            return []
        
        # Use appropriate detection method based on level
        # This is a placeholder - actual implementation depends on detection service
        try:
            if hasattr(self.detection_service, 'detect_frame'):
                result = self.detection_service.detect_frame(frame)
                return result.get('detections', [])
            elif hasattr(self.detection_service, 'detect'):
                return self.detection_service.detect(frame)
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        return []
    
    def _tracked_to_candidates(
        self,
        tracked: List[TrackedObject],
        frame: np.ndarray
    ) -> List[DetectionCandidate]:
        """Convert tracked objects to detection candidates."""
        candidates = []
        
        for obj in tracked:
            # Check if we can use cached OCR
            is_cached = False
            plate_text = obj.plate_text
            ocr_conf = obj.ocr_confidence
            
            if self.config.skip_ocr_for_tracked and obj.plate_text:
                cached = self.tracker.get_cached_ocr(obj.track_id)
                if cached and not self.tracker.should_run_ocr(obj.track_id):
                    plate_text, ocr_conf = cached
                    is_cached = True
            
            candidate = DetectionCandidate(
                bbox=obj.bbox,
                confidence=obj.confidence,
                class_name=obj.class_name,
                plate_text=plate_text,
                ocr_confidence=ocr_conf,
                track_id=obj.track_id,
                is_cached=is_cached
            )
            candidates.append(candidate)
        
        return candidates
    
    def _raw_to_candidates(self, detections: List[Dict]) -> List[DetectionCandidate]:
        """Convert raw detections to candidates."""
        candidates = []
        
        for det in detections:
            candidate = DetectionCandidate(
                bbox=det.get('bbox', (0, 0, 0, 0)),
                confidence=det.get('confidence', 0.0),
                class_name=det.get('class_name', 'plate'),
                plate_text=det.get('plate_text'),
                ocr_confidence=det.get('ocr_confidence', 0.0)
            )
            candidates.append(candidate)
        
        return candidates
    
    def _process_candidate(
        self,
        candidate: DetectionCandidate,
        frame: np.ndarray,
        level: ProcessingLevel
    ) -> DetectionCandidate:
        """
        Process a single detection candidate (OCR, validation).
        """
        # Skip if cached
        if candidate.is_cached:
            return candidate
        
        # Run OCR if not already done and we have the detection service
        if not candidate.plate_text and self.detection_service:
            # Crop plate region
            x1, y1, x2, y2 = candidate.bbox
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                # Run OCR through detection service
                # This is a placeholder - actual implementation depends on service
                pass
        
        # Validate plate text
        if candidate.plate_text and self.config.enable_plate_validation:
            validation = process_ocr_result(candidate.plate_text)
            candidate.validation_result = validation
            
            # Use corrected text
            if validation.is_valid:
                candidate.plate_text = validation.corrected_text
                # Adjust confidence based on validation
                candidate.ocr_confidence = min(
                    1.0,
                    candidate.ocr_confidence + validation.confidence_boost
                )
        
        # Cache OCR result in tracker
        if self.tracker and candidate.track_id and candidate.plate_text:
            self.tracker.cache_ocr_result(
                candidate.track_id,
                candidate.plate_text,
                candidate.ocr_confidence
            )
        
        candidate.processing_level = level
        return candidate
    
    def _apply_smart_zoom(
        self,
        detections: List[DetectionCandidate],
        frame: np.ndarray,
        level: ProcessingLevel
    ) -> List[DetectionCandidate]:
        """
        Re-detect low-confidence plates with zoomed crop.
        """
        if level == ProcessingLevel.INTENSIVE:
            # Already at max processing, skip
            return detections
        
        enhanced = []
        for det in detections:
            # Check if needs smart zoom
            needs_zoom = (
                det.ocr_confidence < self.config.medium_confidence_threshold and
                not det.is_cached
            )
            
            if needs_zoom:
                zoomed_det = self._smart_zoom_detection(det, frame)
                if zoomed_det and zoomed_det.ocr_confidence > det.ocr_confidence:
                    enhanced.append(zoomed_det)
                else:
                    enhanced.append(det)
            else:
                enhanced.append(det)
        
        return enhanced
    
    def _smart_zoom_detection(
        self,
        candidate: DetectionCandidate,
        frame: np.ndarray
    ) -> Optional[DetectionCandidate]:
        """
        Zoom into plate region and re-detect.
        """
        x1, y1, x2, y2 = candidate.bbox
        w, h = x2 - x1, y2 - y1
        
        # Skip if already large enough
        if w > 200 or h > 100:
            return None
        
        # Expand bbox and crop
        expand = int(max(w, h) * 0.3)
        img_h, img_w = frame.shape[:2]
        
        crop_x1 = max(0, x1 - expand)
        crop_y1 = max(0, y1 - expand)
        crop_x2 = min(img_w, x2 + expand)
        crop_y2 = min(img_h, y2 + expand)
        
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Zoom
        zoom = self.config.zoom_factor
        zoomed = cv2.resize(
            crop,
            None,
            fx=zoom,
            fy=zoom,
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply quality enhancement
        if self.quality_processor:
            zoomed = self.quality_processor.enhance_image(zoomed)
        
        # Re-run detection
        # This would call the detection service again
        # For now, return None to skip
        # Actual implementation would need access to OCR service
        
        return None
    
    def _get_previous_avg_confidence(self) -> Optional[float]:
        """Get average confidence from previous frame."""
        if not self._last_detections:
            return None
        
        confidences = [d.confidence for d in self._last_detections]
        return sum(confidences) / len(confidences) if confidences else None
    
    def reset(self):
        """Reset pipeline state."""
        if self.tracker:
            self.tracker.reset()
        self._last_detections = []
        self.frame_count = 0


# Global pipeline instance
_pipeline: Optional[ConfidenceDrivenPipeline] = None


def get_pipeline(
    config: Optional[PipelineConfig] = None,
    detection_service: Any = None
) -> ConfidenceDrivenPipeline:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ConfidenceDrivenPipeline(config, detection_service)
    return _pipeline


def reset_pipeline():
    """Reset the global pipeline."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.reset()
