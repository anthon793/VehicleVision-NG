"""
Video Processing Service using Roboflow Workflow.
Calls ngplates workflow which handles plate detection + Google Vision OCR in one call.
Enhanced with vehicle detection, type classification, and color detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Generator, Optional, Any, Set, Tuple
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from config import settings

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
    vehicle_colors: List[str] = field(default_factory=list)


class WorkflowVideoProcessor:
    """
    Video processor with complete vehicle identification pipeline.
    
    Pipeline:
    1. YOLOv8 (COCO) → Vehicle detection + type classification
    2. Roboflow Workflow → Plate detection + OCR
    3. HSV/K-Means → Vehicle color detection
    4. Enhanced annotations with vehicle box + plate box + labels
    
    OPTIMIZATION: Uses higher frame skip for video to reduce API calls.
    """
    
    def __init__(self, frame_skip: int = None):
        self.frame_skip = frame_skip or settings.FRAME_SKIP
        self.workflow_service = None
        self.detection_service = None  # YOLOv8 vehicle detection
        self.color_service = None      # Vehicle color detection
        # Track seen plates to avoid duplicates
        self.seen_plates: set = set()
    
    def _init_services(self):
        """Lazy initialization of services."""
        if self.workflow_service is None:
            from services.roboflow_workflow import get_workflow_service
            self.workflow_service = get_workflow_service()
        
        if self.detection_service is None:
            from services.detection import get_detection_service
            self.detection_service = get_detection_service()
        
        if self.color_service is None:
            from services.color_detection import get_color_service
            self.color_service = get_color_service()

    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get metadata about a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS)))
        }
        cap.release()
        return info
    
    def extract_frames(self, video_path: str, frame_skip: int = None) -> Generator[tuple, None, None]:
        """Generator to extract frames from video."""
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
                    timestamp_ms = (frame_number / max(1, fps)) * 1000
                    yield frame_number, timestamp_ms, frame
                
                frame_number += 1
        finally:
            cap.release()
    
    def process_frame(self, frame: np.ndarray, stolen_plates: Set[str] = None, fast_mode: bool = False) -> tuple:
        """
        Process a single frame with complete vehicle identification pipeline.
        
        Pipeline:
        1. Roboflow Workflow → Plate detection + OCR on FULL frame (best accuracy)
        2. YOLOv8 (COCO) → Detect vehicles and classify type
        3. Match plates to vehicles based on overlap
        4. Color detection for each vehicle
        
        Args:
            frame: Video frame (BGR format)
            stolen_plates: Set of stolen plate numbers
            fast_mode: If True, uses single API call without retry (for video)
        
        Returns:
            Tuple of (detections, plate_texts, match_statuses, vehicle_colors)
        """
        self._init_services()
        stolen_plates = stolen_plates or set()
        
        # Step 1: Detect plates on FULL FRAME using Roboflow (best OCR accuracy)
        if fast_mode:
            plates, _ = self.workflow_service.detect_plates_fast(frame)
        else:
            plates, _ = self.workflow_service.detect_plates(frame)
        
        # Step 2: Detect vehicles using YOLOv8 COCO
        vehicles = self.detection_service.detect_vehicles(frame)
        
        # Build detection results
        detections = []
        plate_texts = []
        match_statuses = []
        vehicle_colors = []
        
        # Step 3: Match plates to vehicles and build combined detections
        if plates:
            for plate in plates:
                px1, py1, px2, py2 = plate.bbox
                plate_center_x = (px1 + px2) // 2
                plate_center_y = (py1 + py2) // 2
                plate_width = px2 - px1
                plate_height = py2 - py1
                
                # Find the vehicle that contains this plate
                matched_vehicle = None
                vehicle_crop = None
                
                for vehicle in vehicles:
                    vx1, vy1, vx2, vy2 = vehicle.bbox
                    if vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2:
                        matched_vehicle = vehicle
                        vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if vy2 > vy1 and vx2 > vx1 else None
                        break
                
                # If no vehicle matched, estimate vehicle region around the plate
                # Plates are typically at bottom of vehicle, estimate body above it
                vehicle_color = "unknown"
                vehicle_type = "car"  # Default assumption
                vehicle_class_id = 2   # COCO car class
                
                if matched_vehicle:
                    vehicle_bbox = matched_vehicle.bbox
                    vehicle_type = matched_vehicle.class_name
                    vehicle_confidence = matched_vehicle.confidence
                    vehicle_class_id = matched_vehicle.class_id
                    
                    if vehicle_crop is not None and vehicle_crop.size > 0:
                        vehicle_color, _ = self.color_service.detect_vehicle_color(vehicle_crop)
                else:
                    # Estimate vehicle region: expand around plate
                    # Plates are usually at car front/back, vehicle body is above
                    h, w = frame.shape[:2]
                    expand_x = max(plate_width * 3, 150)  # Expand horizontally
                    expand_up = max(plate_height * 8, 200)  # Expand more upward (body above plate)
                    expand_down = max(plate_height * 2, 50)  # Less expansion downward
                    
                    est_x1 = max(0, px1 - expand_x // 2)
                    est_x2 = min(w, px2 + expand_x // 2)
                    est_y1 = max(0, py1 - expand_up)
                    est_y2 = min(h, py2 + expand_down)
                    
                    vehicle_bbox = (est_x1, est_y1, est_x2, est_y2)
                    vehicle_confidence = plate.confidence * 0.8
                    
                    # Get color from estimated region
                    vehicle_crop = frame[est_y1:est_y2, est_x1:est_x2].copy()
                    if vehicle_crop is not None and vehicle_crop.size > 0:
                        vehicle_color, _ = self.color_service.detect_vehicle_color(vehicle_crop)
                
                # Check if plate is stolen
                plate_text = plate.plate_text
                normalized_text = plate_text.upper().replace(" ", "").replace("-", "") if plate_text else ""
                is_stolen = normalized_text in stolen_plates if normalized_text else False
                
                # Build detection dict
                detection = {
                    "vehicle": {
                        "bbox": vehicle_bbox,
                        "confidence": vehicle_confidence,
                        "class_id": vehicle_class_id,
                        "class_name": vehicle_type
                    },
                    "vehicle_type": vehicle_type,
                    "vehicle_confidence": vehicle_confidence,
                    "vehicle_color": vehicle_color,
                    "cropped_image": vehicle_crop,
                    "plates": [{
                        "bbox": plate.bbox,
                        "local_bbox": plate.bbox,
                        "confidence": plate.confidence,
                        "cropped_image": frame[py1:py2, px1:px2].copy() if py2 > py1 and px2 > px1 else None
                    }]
                }
                detections.append(detection)
                plate_texts.append(plate_text)
                match_statuses.append(is_stolen)
                vehicle_colors.append(vehicle_color)
        
        elif vehicles:
            # No plates found, but vehicles detected - show vehicles with color
            for vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle.bbox
                vehicle_crop = frame[vy1:vy2, vx1:vx2].copy() if vy2 > vy1 and vx2 > vx1 else None
                
                vehicle_color = "unknown"
                if vehicle_crop is not None and vehicle_crop.size > 0:
                    vehicle_color, _ = self.color_service.detect_vehicle_color(vehicle_crop)
                
                detection = {
                    "vehicle": {
                        "bbox": vehicle.bbox,
                        "confidence": vehicle.confidence,
                        "class_id": vehicle.class_id,
                        "class_name": vehicle.class_name
                    },
                    "vehicle_type": vehicle.class_name,
                    "vehicle_confidence": vehicle.confidence,
                    "vehicle_color": vehicle_color,
                    "cropped_image": vehicle_crop,
                    "plates": []
                }
                detections.append(detection)
                plate_texts.append("")
                match_statuses.append(False)
                vehicle_colors.append(vehicle_color)
        
        return detections, plate_texts, match_statuses, vehicle_colors
    
    def process_image(self, image: np.ndarray, stolen_plates: Set[str] = None) -> Dict[str, Any]:
        """
        Process a single image with complete vehicle identification.
        
        Returns:
            Dict with detection results, vehicle colors, and annotated image
        """
        self._init_services()
        stolen_plates = stolen_plates or set()
        start_time = time.time()
        
        # Get detections via enhanced pipeline
        detections, plate_texts, match_statuses, vehicle_colors = self.process_frame(image, stolen_plates)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Draw enhanced annotations with vehicle box + plate box + labels
        annotated_image = self._draw_annotations(
            image, detections, plate_texts, match_statuses, vehicle_colors
        )
        
        return {
            "detections": detections,
            "plate_texts": plate_texts,
            "match_statuses": match_statuses,
            "vehicle_colors": vehicle_colors,
            "processing_time_ms": processing_time,
            "annotated_image": annotated_image
        }
    
    def _draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict],
        plate_texts: List[str],
        match_statuses: List[bool],
        vehicle_colors: List[str] = None
    ) -> np.ndarray:
        """
        Draw enhanced bounding boxes and labels on image.
        
        Draws:
        - Large vehicle bounding box (blue/red if stolen)
        - Smaller plate bounding box (green/red if stolen)
        - Combined label: Vehicle: <Type> | Color: <Color> | Plate: <Number>
        """
        output = image.copy()
        vehicle_colors = vehicle_colors or ["unknown"] * len(detections)
        
        for i, (detection, text, is_stolen, color) in enumerate(
            zip(detections, plate_texts, match_statuses, vehicle_colors)
        ):
            vehicle_info = detection.get("vehicle", {})
            vehicle_type = detection.get("vehicle_type", "vehicle")
            vehicle_color = detection.get("vehicle_color", color)
            plates = detection.get("plates", [])
            
            # Get vehicle bounding box
            vx1, vy1, vx2, vy2 = vehicle_info.get("bbox", (0, 0, 0, 0))
            
            # Color scheme: red for stolen, blue for vehicle box
            vehicle_box_color = (0, 0, 255) if is_stolen else (255, 150, 0)  # Orange/Red
            plate_box_color = (0, 0, 255) if is_stolen else (0, 255, 0)      # Red/Green
            
            # Draw vehicle bounding box (thick line)
            if vx2 > vx1 and vy2 > vy1:
                cv2.rectangle(output, (vx1, vy1), (vx2, vy2), vehicle_box_color, 3)
            
            # Draw plate bounding box (inside vehicle)
            if plates:
                plate = plates[0]
                px1, py1, px2, py2 = plate.get("bbox", (0, 0, 0, 0))
                if px2 > px1 and py2 > py1:
                    cv2.rectangle(output, (px1, py1), (px2, py2), plate_box_color, 2)
            
            # Build combined label
            type_str = vehicle_type.capitalize() if vehicle_type else "Vehicle"
            color_str = vehicle_color.capitalize() if vehicle_color and vehicle_color != "unknown" else "?"
            plate_str = text if text else "?"
            
            if is_stolen:
                label = f"STOLEN | {type_str} | {color_str} | {plate_str}"
            else:
                label = f"{type_str} | Color: {color_str} | Plate: {plate_str}"
            
            # Calculate label position (above vehicle box)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            label_x = vx1 if vx1 > 0 else 10
            label_y = vy1 - 10 if vy1 > 30 else vy2 + text_height + 10
            
            # Draw label background
            bg_color = (0, 0, 200) if is_stolen else (50, 50, 50)  # Dark red / Dark gray
            cv2.rectangle(
                output,
                (label_x, label_y - text_height - 8),
                (label_x + text_width + 16, label_y + 4),
                bg_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output, label, (label_x + 8, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # If stolen, add prominent warning
            if is_stolen:
                # Draw flashing-style border
                cv2.rectangle(output, (vx1-2, vy1-2), (vx2+2, vy2+2), (0, 0, 255), 4)
        
        return output
    
    def process_video(
        self,
        video_path: str,
        stolen_plates: Set[str] = None,
        progress_callback: callable = None
    ) -> List[FrameResult]:
        """
        Process entire video file.
        
        OPTIMIZED:
        - Processes ~1 frame/second (frame_skip=30 for 30fps)
        - Skips duplicate plate detection  
        - Tracks unique plates found
        """
        self._init_services()
        
        results = []
        self.seen_plates = set()  # Reset for new video
        unique_plates_found = {}  # plate -> first detection info
        
        video_info = self.get_video_info(video_path)
        total_frames = video_info["frame_count"]
        fps = video_info['fps']
        duration = video_info['duration_seconds']
        
        # Calculate frames to process
        frames_to_process = total_frames // (self.frame_skip + 1)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Duration: {duration}s, FPS: {fps}, Total frames: {total_frames}")
        logger.info(f"Frame skip: {self.frame_skip} -> Processing ~{frames_to_process} frames")
        
        processed_count = 0
        
        for frame_number, timestamp_ms, frame in self.extract_frames(video_path):
            start_time = time.time()
            
            # Use fast_mode=True for video (single API call, no retry)
            detections, plate_texts, match_statuses, vehicle_colors = self.process_frame(frame, stolen_plates, fast_mode=True)
            
            processing_time = (time.time() - start_time) * 1000
            processed_count += 1
            
            # Filter out already-seen plates (to avoid duplicates in results)
            new_detections = []
            new_texts = []
            new_statuses = []
            new_colors = []
            
            for det, text, status, color in zip(detections, plate_texts, match_statuses, vehicle_colors):
                normalized = text.upper().replace(' ', '').replace('-', '') if text else ''
                if normalized and normalized not in self.seen_plates:
                    self.seen_plates.add(normalized)
                    new_detections.append(det)
                    new_texts.append(text)
                    new_statuses.append(status)
                    new_colors.append(color)
                    unique_plates_found[normalized] = {
                        'text': text,
                        'frame': frame_number,
                        'timestamp': timestamp_ms,
                        'is_stolen': status,
                        'vehicle_color': color
                    }
                    logger.info(f"New plate found at {timestamp_ms/1000:.1f}s: {text} ({color})")
            
            if new_detections:
                results.append(FrameResult(
                    frame_number=frame_number,
                    timestamp_ms=timestamp_ms,
                    detections=new_detections,
                    plate_texts=new_texts,
                    match_statuses=new_statuses,
                    processing_time_ms=processing_time,
                    vehicle_colors=new_colors
                ))
            
            if progress_callback:
                progress_callback(processed_count, frames_to_process)
        
        logger.info(f"Video processing complete.")
        logger.info(f"  Frames processed: {processed_count}")
        logger.info(f"  Unique plates found: {len(unique_plates_found)}")
        for plate, info in unique_plates_found.items():
            status = "STOLEN!" if info['is_stolen'] else "ok"
            logger.info(f"    - {info['text']} at {info['timestamp']/1000:.1f}s [{status}]")
        
        return results

    def process_video_streaming(
        self,
        video_path: str,
        stolen_plates: Set[str] = None
    ) -> Generator[FrameResult, None, None]:
        """
        Process video file with streaming results (yields each frame result).
        
        Used for WebSocket real-time updates.
        
        Args:
            video_path: Path to video file
            stolen_plates: Set of stolen plate numbers to check against
            
        Yields:
            FrameResult for each processed frame
        """
        self._init_services()
        self.seen_plates = set()  # Reset for new video
        
        video_info = self.get_video_info(video_path)
        total_frames = video_info["frame_count"]
        fps = video_info['fps']
        
        logger.info(f"Streaming video processing: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        for frame_number, timestamp_ms, frame in self.extract_frames(video_path):
            start_time = time.time()
            
            # Use fast_mode=True for video
            detections, plate_texts, match_statuses, vehicle_colors = self.process_frame(frame, stolen_plates, fast_mode=True)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Yield result for every frame (even if no detections)
            yield FrameResult(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                detections=detections,
                plate_texts=plate_texts,
                match_statuses=match_statuses,
                processing_time_ms=processing_time,
                vehicle_colors=vehicle_colors
            )


# Singleton instance
_workflow_processor: Optional[WorkflowVideoProcessor] = None


def get_workflow_processor() -> WorkflowVideoProcessor:
    """Get or create singleton workflow processor."""
    global _workflow_processor
    if _workflow_processor is None:
        _workflow_processor = WorkflowVideoProcessor()
    return _workflow_processor


# For backward compatibility, also export as get_video_processor
def get_video_processor() -> WorkflowVideoProcessor:
    """Backward compatible alias."""
    return get_workflow_processor()
