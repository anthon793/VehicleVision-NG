"""
Vehicle Color Detection Service.
Extracts dominant color from vehicle images using HSV analysis and K-means clustering.
Improved with better handling of dark colors (black, dark gray) and real-world lighting.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# Common vehicle colors for validation
COMMON_VEHICLE_COLORS = [
    "white", "black", "silver", "gray", "red", "blue", 
    "brown", "green", "beige", "orange", "yellow", "maroon"
]


class ColorDetectionService:
    """
    Service for detecting dominant vehicle colors.
    Uses improved HSV analysis prioritizing brightness-based detection for dark colors.
    """
    
    def __init__(self, n_clusters: int = 5):
        """
        Initialize color detection service.
        
        Args:
            n_clusters: Number of clusters for K-means (more = finer detection)
        """
        self.n_clusters = n_clusters
    
    def detect_vehicle_color(self, vehicle_image: np.ndarray) -> Tuple[str, float]:
        """
        Detect the dominant color of a vehicle from its cropped image.
        
        Args:
            vehicle_image: Cropped vehicle image (BGR format from OpenCV)
            
        Returns:
            Tuple of (color_name, confidence_score)
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return "unknown", 0.0
        
        try:
            # Focus on the center/body of the vehicle (avoid wheels, windows, background)
            height, width = vehicle_image.shape[:2]
            
            # Crop to center region (excluding top 35% for windows, bottom 25% for wheels)
            y_start = int(height * 0.35)
            y_end = int(height * 0.75)
            x_start = int(width * 0.20)
            x_end = int(width * 0.80)
            
            body_region = vehicle_image[y_start:y_end, x_start:x_end]
            
            if body_region.size == 0 or body_region.shape[0] < 10 or body_region.shape[1] < 10:
                body_region = vehicle_image
            
            # Convert to HSV for color analysis
            hsv_image = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
            
            # Get dominant color using improved method
            color_name, confidence = self._detect_dominant_color(hsv_image, body_region)
            
            logger.info(f"Detected vehicle color: {color_name} (confidence: {confidence:.2f})")
            return color_name, confidence
            
        except Exception as e:
            logger.error(f"Color detection error: {str(e)}")
            return "unknown", 0.0
    
    def _detect_dominant_color(self, hsv_image: np.ndarray, bgr_image: np.ndarray) -> Tuple[str, float]:
        """
        Detect dominant color using HSV analysis with priority to brightness-based detection.
        
        The key insight: check brightness (V) first to detect black/dark gray,
        then check saturation (S) for white/silver/gray, then hue (H) for chromatic colors.
        """
        # Flatten and analyze pixel distribution
        h_channel = hsv_image[:, :, 0].flatten()
        s_channel = hsv_image[:, :, 1].flatten()
        v_channel = hsv_image[:, :, 2].flatten()
        
        total_pixels = len(v_channel)
        
        # Step 1: Check for DARK colors first (based on Value/brightness)
        # Dark pixels: V < 60 (very dark) or V < 100 with low saturation
        very_dark_mask = v_channel < 60
        dark_mask = (v_channel < 100) & (s_channel < 80)
        
        very_dark_ratio = np.sum(very_dark_mask) / total_pixels
        dark_ratio = np.sum(dark_mask) / total_pixels
        
        # If majority of pixels are dark -> black or dark gray
        if very_dark_ratio > 0.35:
            return "black", min(0.95, very_dark_ratio + 0.2)
        
        if dark_ratio > 0.40:
            # Check if it's black or dark gray
            avg_v = np.mean(v_channel[dark_mask]) if np.any(dark_mask) else 0
            if avg_v < 70:
                return "black", min(0.9, dark_ratio + 0.1)
            else:
                return "gray", min(0.85, dark_ratio)
        
        # Step 2: Check for LIGHT/ACHROMATIC colors (based on Saturation)
        # Low saturation = white, silver, or gray
        low_sat_mask = s_channel < 50
        low_sat_ratio = np.sum(low_sat_mask) / total_pixels
        
        if low_sat_ratio > 0.45:
            avg_v = np.mean(v_channel[low_sat_mask])
            if avg_v > 200:
                return "white", min(0.9, low_sat_ratio)
            elif avg_v > 150:
                return "silver", min(0.85, low_sat_ratio)
            elif avg_v > 80:
                return "gray", min(0.85, low_sat_ratio)
            else:
                return "black", min(0.85, low_sat_ratio)
        
        # Step 3: Chromatic color detection (based on Hue)
        # Only consider pixels with decent saturation and value
        chromatic_mask = (s_channel >= 40) & (v_channel >= 50)
        
        if np.sum(chromatic_mask) < total_pixels * 0.1:
            # Not enough chromatic pixels, fallback to K-means
            return self._classify_color_kmeans(bgr_image)
        
        chromatic_hues = h_channel[chromatic_mask]
        chromatic_sats = s_channel[chromatic_mask]
        chromatic_vals = v_channel[chromatic_mask]
        
        # Count colors by hue ranges
        color_counts = {
            "red": 0,
            "maroon": 0,
            "orange": 0,
            "yellow": 0,
            "green": 0,
            "blue": 0,
            "purple": 0,
            "pink": 0,
            "brown": 0,
        }
        
        for h, s, v in zip(chromatic_hues, chromatic_sats, chromatic_vals):
            # Red wraps around 0/180
            if h < 8 or h >= 165:
                if v < 100:
                    color_counts["maroon"] += 1
                else:
                    color_counts["red"] += 1
            elif h < 20:
                if v < 120 and s > 100:
                    color_counts["brown"] += 1
                else:
                    color_counts["orange"] += 1
            elif h < 35:
                color_counts["yellow"] += 1
            elif h < 80:
                color_counts["green"] += 1
            elif h < 130:
                color_counts["blue"] += 1
            elif h < 150:
                color_counts["purple"] += 1
            else:
                color_counts["pink"] += 1
        
        if max(color_counts.values()) == 0:
            return self._classify_color_kmeans(bgr_image)
        
        dominant_color = max(color_counts, key=color_counts.get)
        confidence = color_counts[dominant_color] / len(chromatic_hues)
        
        # Maroon is often called "red" colloquially
        if dominant_color == "maroon":
            dominant_color = "red"  # or keep as maroon if you prefer
        
        return dominant_color, min(0.95, confidence + 0.2)
    
    def _classify_color_kmeans(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify color using K-means clustering on the image.
        Fallback method when HSV analysis is inconclusive.
        """
        try:
            # Resize for faster processing
            small_image = cv2.resize(image, (80, 80))
            
            # Reshape to list of pixels
            pixels = small_image.reshape(-1, 3).astype(np.float32)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels, self.n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Count pixels in each cluster
            label_counts = Counter(labels.flatten())
            
            # Get the dominant cluster
            dominant_label = label_counts.most_common(1)[0][0]
            dominant_count = label_counts[dominant_label]
            dominant_color_bgr = centers[dominant_label].astype(int)
            
            # Convert to color name
            color_name = self._bgr_to_color_name(dominant_color_bgr)
            confidence = dominant_count / len(labels)
            
            return color_name, min(0.9, confidence + 0.1)
            
        except Exception as e:
            logger.error(f"K-means color detection error: {str(e)}")
            return "unknown", 0.0
    
    def _bgr_to_color_name(self, bgr: np.ndarray) -> str:
        """
        Convert BGR color value to a color name.
        
        Args:
            bgr: BGR color value [B, G, R]
            
        Returns:
            Color name string
        """
        # Convert to HSV
        bgr_pixel = np.uint8([[bgr]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0, 0]
        
        # PRIORITY 1: Check brightness first for dark colors
        if v < 50:
            return "black"
        if v < 80 and s < 60:
            return "black"
        
        # PRIORITY 2: Low saturation = achromatic colors
        if s < 40:
            if v < 100:
                return "gray"
            elif v < 180:
                return "silver"
            else:
                return "white"
        
        # PRIORITY 3: Chromatic colors based on hue
        if h < 8 or h >= 165:
            return "red" if v > 100 else "maroon"
        elif h < 22:
            return "brown" if v < 120 else "orange"
        elif h < 38:
            return "yellow"
        elif h < 80:
            return "green"
        elif h < 130:
            return "blue"
        elif h < 150:
            return "purple"
        else:
            return "pink"
    
    def get_color_hex(self, color_name: str) -> str:
        """
        Get hex color code for a color name (for UI display).
        """
        color_hex_map = {
            "red": "#FF0000",
            "maroon": "#800000",
            "orange": "#FFA500",
            "yellow": "#FFFF00",
            "green": "#008000",
            "blue": "#0000FF",
            "purple": "#800080",
            "pink": "#FFC0CB",
            "white": "#FFFFFF",
            "silver": "#C0C0C0",
            "gray": "#808080",
            "black": "#000000",
            "brown": "#8B4513",
            "beige": "#F5F5DC",
            "unknown": "#CCCCCC"
        }
        return color_hex_map.get(color_name, "#CCCCCC")


# Singleton instance
_color_service: Optional[ColorDetectionService] = None


def get_color_service() -> ColorDetectionService:
    """
    Get or create singleton color detection service instance.
    """
    global _color_service
    if _color_service is None:
        _color_service = ColorDetectionService()
    return _color_service

