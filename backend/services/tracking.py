"""
Object Tracking Service using SORT (Simple Online and Realtime Tracking).
Maintains vehicle/plate identity across frames to reduce redundant OCR.

Academic Justification (Chapter 4):
- Multi-frame consistency reduces false positives
- Avoids re-running expensive OCR on the same tracked object
- Provides stable bounding boxes for better plate crops
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    plate_text: Optional[str] = None
    ocr_confidence: float = 0.0
    frames_tracked: int = 1
    frames_since_seen: int = 0
    last_seen_frame: int = 0
    ocr_results: List[str] = field(default_factory=list)  # For majority voting
    is_stolen: bool = False


class SORTTracker:
    """
    Simplified SORT tracker implementation.
    Uses IoU-based association without Kalman filtering for simplicity.
    
    Suitable for BSc-level demonstration while providing core tracking benefits.
    """
    
    def __init__(
        self,
        max_age: int = 10,           # Max frames to keep lost tracks
        min_hits: int = 3,            # Min detections before track is confirmed
        iou_threshold: float = 0.3,   # IoU threshold for matching
        ocr_cache_frames: int = 30    # Frames to cache OCR result
    ):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to confirm a track
            iou_threshold: IOU threshold for matching detections to tracks
            ocr_cache_frames: Number of frames to reuse cached OCR result
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.ocr_cache_frames = ocr_cache_frames
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        # OCR cache: track_id -> (plate_text, confidence, frame_cached)
        self.ocr_cache: Dict[int, Tuple[str, float, int]] = {}
        
        logger.info(f"SORT Tracker initialized (max_age={max_age}, min_hits={min_hits})")
    
    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection coordinates
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        # Intersection area
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _match_detections(
        self, 
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            matches: List of (track_id, detection_idx) pairs
            unmatched_tracks: List of unmatched track IDs
            unmatched_detections: List of unmatched detection indices
        """
        if not self.tracks or not detections:
            return [], list(self.tracks.keys()), list(range(len(detections)))
        
        # Build IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                det_bbox = det.get('bbox', (0, 0, 0, 0))
                iou_matrix[i, j] = self._compute_iou(track.bbox, det_bbox)
        
        # Greedy matching
        matches = []
        matched_tracks = set()
        matched_detections = set()
        
        # Sort by IoU (descending) and match greedily
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            track_idx, det_idx = max_idx
            
            track_id = track_ids[track_idx]
            matches.append((track_id, det_idx))
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            
            # Remove matched row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List[Dict]) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_name'
            
        Returns:
            List of confirmed TrackedObject instances
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        matches, unmatched_tracks, unmatched_dets = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, det_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_id]
            
            track.bbox = det.get('bbox', track.bbox)
            track.confidence = det.get('confidence', track.confidence)
            track.frames_tracked += 1
            track.frames_since_seen = 0
            track.last_seen_frame = self.frame_count
            
            # Update plate text if provided (for OCR aggregation)
            if det.get('plate_text'):
                track.ocr_results.append(det['plate_text'])
                # Keep only last 5 results for majority voting
                track.ocr_results = track.ocr_results[-5:]
        
        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id].frames_since_seen += 1
        
        # Remove old tracks
        tracks_to_remove = [
            tid for tid, track in self.tracks.items() 
            if track.frames_since_seen > self.max_age
        ]
        for tid in tracks_to_remove:
            del self.tracks[tid]
            if tid in self.ocr_cache:
                del self.ocr_cache[tid]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = TrackedObject(
                track_id=self.next_track_id,
                bbox=det.get('bbox', (0, 0, 0, 0)),
                confidence=det.get('confidence', 0.0),
                class_name=det.get('class_name', 'vehicle'),
                last_seen_frame=self.frame_count
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Return confirmed tracks (seen enough times)
        confirmed = [
            track for track in self.tracks.values()
            if track.frames_tracked >= self.min_hits and track.frames_since_seen == 0
        ]
        
        return confirmed
    
    def should_run_ocr(self, track_id: int) -> bool:
        """
        Check if OCR should be run for a tracked object.
        Avoids redundant OCR on the same tracked plate.
        
        Args:
            track_id: Track ID to check
            
        Returns:
            True if OCR should be run, False if cached result is valid
        """
        if track_id not in self.ocr_cache:
            return True
        
        _, _, cached_frame = self.ocr_cache[track_id]
        frames_since_cache = self.frame_count - cached_frame
        
        return frames_since_cache > self.ocr_cache_frames
    
    def cache_ocr_result(self, track_id: int, plate_text: str, confidence: float):
        """Cache OCR result for a track."""
        self.ocr_cache[track_id] = (plate_text, confidence, self.frame_count)
        
        if track_id in self.tracks:
            self.tracks[track_id].plate_text = plate_text
            self.tracks[track_id].ocr_confidence = confidence
    
    def get_cached_ocr(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Get cached OCR result for a track."""
        if track_id in self.ocr_cache:
            plate_text, confidence, _ = self.ocr_cache[track_id]
            return plate_text, confidence
        return None
    
    def get_majority_vote_plate(self, track_id: int) -> Optional[str]:
        """
        Get plate text using majority voting from multiple OCR results.
        Improves accuracy through temporal aggregation.
        """
        if track_id not in self.tracks:
            return None
        
        ocr_results = self.tracks[track_id].ocr_results
        if not ocr_results:
            return None
        
        # Count occurrences
        from collections import Counter
        counts = Counter(ocr_results)
        most_common = counts.most_common(1)
        
        if most_common:
            return most_common[0][0]
        return None
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.ocr_cache.clear()
        self.next_track_id = 1
        self.frame_count = 0


# Global tracker instance
_tracker_instance: Optional[SORTTracker] = None


def get_tracker() -> SORTTracker:
    """Get or create global tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SORTTracker()
    return _tracker_instance


def reset_tracker():
    """Reset the global tracker."""
    global _tracker_instance
    if _tracker_instance is not None:
        _tracker_instance.reset()
