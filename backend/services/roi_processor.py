"""
Region of Interest (ROI) Restriction Module.
Limits detection to likely plate zones to reduce false positives and improve speed.

Academic Justification (Chapter 4):
- License plates are typically in lower half of vehicle
- Reduces computational load by cropping irrelevant regions
- Improves detection accuracy by focusing on plate-likely zones
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ROIConfig:
    """Configuration for ROI extraction."""
    # Relative coordinates (0-1 range)
    lower_half_only: bool = True
    lane_centered: bool = True
    lane_width_ratio: float = 0.7  # Width of center lane region
    plate_zone_top: float = 0.4     # Top boundary of plate zone (40% from top)
    plate_zone_bottom: float = 1.0   # Bottom boundary (100% = full bottom)
    min_crop_size: int = 100         # Minimum crop dimension
    enable_roi: bool = True          # Master toggle


class ROIProcessor:
    """
    Processes images to extract Region of Interest (ROI) for plate detection.
    Focuses on areas where license plates are most likely to appear.
    """
    
    def __init__(self, config: Optional[ROIConfig] = None):
        """
        Initialize ROI processor.
        
        Args:
            config: ROI configuration. Uses defaults if not provided.
        """
        self.config = config or ROIConfig()
        logger.info(f"ROI Processor initialized (enabled={self.config.enable_roi})")
    
    def get_roi_coords(
        self, 
        image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate ROI coordinates for an image.
        
        Args:
            image_shape: (height, width, channels) of the image
            
        Returns:
            (x1, y1, x2, y2) ROI coordinates
        """
        h, w = image_shape[:2]
        
        if not self.config.enable_roi:
            return (0, 0, w, h)
        
        # Calculate vertical bounds (plate zone)
        y1 = int(h * self.config.plate_zone_top)
        y2 = int(h * self.config.plate_zone_bottom)
        
        # Calculate horizontal bounds (lane centering)
        if self.config.lane_centered:
            center_x = w // 2
            half_width = int(w * self.config.lane_width_ratio / 2)
            x1 = max(0, center_x - half_width)
            x2 = min(w, center_x + half_width)
        else:
            x1, x2 = 0, w
        
        # Ensure minimum size
        if (x2 - x1) < self.config.min_crop_size:
            x1 = max(0, (w - self.config.min_crop_size) // 2)
            x2 = min(w, x1 + self.config.min_crop_size)
        
        if (y2 - y1) < self.config.min_crop_size:
            y1 = max(0, y2 - self.config.min_crop_size)
        
        return (x1, y1, x2, y2)
    
    def crop_roi(
        self, 
        image: np.ndarray,
        roi_coords: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop image to ROI.
        
        Args:
            image: Input image as numpy array
            roi_coords: Optional specific ROI coords. Calculates if not provided.
            
        Returns:
            (cropped_image, roi_coords)
        """
        if roi_coords is None:
            roi_coords = self.get_roi_coords(image.shape)
        
        x1, y1, x2, y2 = roi_coords
        cropped = image[y1:y2, x1:x2]
        
        return cropped, roi_coords
    
    def adjust_detections_to_original(
        self,
        detections: List[Dict],
        roi_coords: Tuple[int, int, int, int]
    ) -> List[Dict]:
        """
        Adjust detection coordinates from ROI space back to original image space.
        
        Args:
            detections: List of detection dicts with 'bbox' key
            roi_coords: (x1, y1, x2, y2) of the ROI in original image
            
        Returns:
            Detections with adjusted bounding boxes
        """
        roi_x1, roi_y1, _, _ = roi_coords
        
        adjusted = []
        for det in detections:
            det_copy = det.copy()
            if 'bbox' in det_copy:
                x1, y1, x2, y2 = det_copy['bbox']
                det_copy['bbox'] = (
                    x1 + roi_x1,
                    y1 + roi_y1,
                    x2 + roi_x1,
                    y2 + roi_y1
                )
            adjusted.append(det_copy)
        
        return adjusted
    
    def extract_vehicle_roi(
        self,
        image: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        expand_ratio: float = 0.1
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract ROI around a detected vehicle, focusing on plate zone.
        
        Args:
            image: Full image
            vehicle_bbox: (x1, y1, x2, y2) of detected vehicle
            expand_ratio: How much to expand the bbox
            
        Returns:
            (cropped_vehicle_roi, adjusted_coords)
        """
        h, w = image.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_bbox
        
        # Expand bbox slightly
        v_w = vx2 - vx1
        v_h = vy2 - vy1
        expand_x = int(v_w * expand_ratio)
        expand_y = int(v_h * expand_ratio)
        
        # Focus on lower portion of vehicle (where plate usually is)
        plate_top = vy1 + int(v_h * 0.5)  # Lower 50% of vehicle
        
        # Final coords with expansion
        x1 = max(0, vx1 - expand_x)
        y1 = max(0, plate_top)
        x2 = min(w, vx2 + expand_x)
        y2 = min(h, vy2 + expand_y)
        
        cropped = image[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2, y2)


def get_multi_scale_rois(
    image: np.ndarray,
    scales: List[float] = [1.0, 0.75, 0.5]
) -> List[Tuple[np.ndarray, float]]:
    """
    Generate multiple ROIs at different scales for detection.
    Useful for detecting plates at various distances.
    
    Args:
        image: Input image
        scales: List of scale factors
        
    Returns:
        List of (scaled_image, scale_factor) tuples
    """
    results = []
    
    for scale in scales:
        if scale == 1.0:
            results.append((image, 1.0))
        else:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            results.append((scaled, scale))
    
    return results


def merge_multi_scale_detections(
    all_detections: List[Tuple[List[Dict], float]],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Merge detections from multiple scales using NMS.
    
    Args:
        all_detections: List of (detections, scale_factor) tuples
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Merged list of detections
    """
    # Adjust all detections to original scale
    adjusted = []
    for detections, scale in all_detections:
        for det in detections:
            det_copy = det.copy()
            if 'bbox' in det_copy and scale != 1.0:
                x1, y1, x2, y2 = det_copy['bbox']
                det_copy['bbox'] = (
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale)
                )
            adjusted.append(det_copy)
    
    # Apply NMS
    if not adjusted:
        return []
    
    boxes = np.array([d['bbox'] for d in adjusted])
    scores = np.array([d.get('confidence', 0.5) for d in adjusted])
    
    # Simple NMS implementation
    keep = []
    order = scores.argsort()[::-1]
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = []
        for j in remaining:
            iou = _compute_iou(boxes[i], boxes[j])
            ious.append(iou)
        
        ious = np.array(ious)
        order = remaining[ious < iou_threshold]
    
    return [adjusted[i] for i in keep]


def _compute_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


# Global ROI processor instance
_roi_processor: Optional[ROIProcessor] = None


def get_roi_processor(config: Optional[ROIConfig] = None) -> ROIProcessor:
    """Get or create global ROI processor."""
    global _roi_processor
    if _roi_processor is None:
        _roi_processor = ROIProcessor(config)
    return _roi_processor
