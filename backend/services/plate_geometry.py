"""
Plate Geometry Correction Module.
Handles perspective correction, rotation detection, and deskewing for angled license plates.

This module improves OCR accuracy by:
1. Detecting plate orientation and perspective distortion
2. Applying homography transformation to correct perspective
3. Handling multi-angle rotation for slanted plates
4. Pre-processing distorted text regions

Academic Justification:
- Perspective distortion causes OCR character misalignment
- Rotation causes character segmentation errors
- Geometric correction significantly improves recognition accuracy
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class PlateGeometry:
    """Detected plate geometry parameters."""
    angle: float              # Rotation angle in degrees
    skew: float               # Horizontal skew angle
    perspective_distortion: float  # Degree of perspective distortion (0-1)
    corners: Optional[np.ndarray] = None  # Four corner points if detected
    is_correctable: bool = True
    
    @property
    def rotation_angle(self) -> float:
        """Alias for angle (rotation)."""
        return self.angle
    
    @property
    def skew_angle(self) -> float:
        """Alias for skew."""
        return self.skew
    
    @property
    def total_angle(self) -> float:
        """Combined rotation and skew angle."""
        return self.angle + self.skew
    
    @property
    def needs_correction(self) -> bool:
        """Check if significant correction is needed."""
        return abs(self.total_angle) > 2.0 or self.perspective_distortion > 0.1


@dataclass
class CorrectionConfig:
    """Configuration for plate geometry correction."""
    enable_perspective_correction: bool = True
    enable_rotation_correction: bool = True
    enable_deskew: bool = True
    max_rotation_angle: float = 45.0  # Max angle to attempt correction
    min_plate_area: int = 500  # Minimum plate area in pixels
    perspective_threshold: float = 0.15  # Threshold to apply perspective correction
    use_multi_angle_ocr: bool = True  # Try multiple angles if first fails
    rotation_angles: List[float] = None  # Angles to try for multi-angle OCR
    
    def __post_init__(self):
        if self.rotation_angles is None:
            self.rotation_angles = [-15, -10, -5, 0, 5, 10, 15]


class PlateGeometryCorrector:
    """
    Corrects perspective distortion and rotation in license plate images.
    Improves OCR accuracy for slanted/angled plates.
    """
    
    def __init__(self, config: Optional[CorrectionConfig] = None):
        """Initialize the geometry corrector."""
        self.config = config or CorrectionConfig()
        logger.info("Plate Geometry Corrector initialized")
    
    def analyze_plate_geometry(self, plate_image: np.ndarray) -> PlateGeometry:
        """
        Analyze plate image to detect rotation, skew, and perspective distortion.
        
        Args:
            plate_image: Cropped plate image (BGR or grayscale)
            
        Returns:
            PlateGeometry with detected parameters
        """
        if plate_image is None or plate_image.size == 0:
            return PlateGeometry(angle=0, skew=0, perspective_distortion=0, is_correctable=False)
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Detect rotation angle using multiple methods
        angle = self._detect_rotation_angle(gray)
        
        # Detect skew (horizontal tilt)
        skew = self._detect_skew(gray)
        
        # Detect perspective distortion
        perspective_distortion, corners = self._detect_perspective_distortion(gray)
        
        # Determine if correction is feasible
        is_correctable = (
            abs(angle) <= self.config.max_rotation_angle and
            plate_image.shape[0] * plate_image.shape[1] >= self.config.min_plate_area
        )
        
        return PlateGeometry(
            angle=angle,
            skew=skew,
            perspective_distortion=perspective_distortion,
            corners=corners,
            is_correctable=is_correctable
        )
    
    def _detect_rotation_angle(self, gray: np.ndarray) -> float:
        """
        Detect rotation angle using Hough lines and edge analysis.
        """
        # Method 1: Hough Line Transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                 minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Only consider nearly horizontal lines (plate text direction)
                    if abs(angle) < 45:
                        angles.append(angle)
            
            if angles:
                # Use median to be robust against outliers
                return np.median(angles)
        
        # Method 2: Minimum area rectangle
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                # Normalize angle
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                return angle
        
        return 0.0
    
    def _detect_skew(self, gray: np.ndarray) -> float:
        """
        Detect horizontal skew angle using projection profile analysis.
        """
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try different angles and find the one with max horizontal projection variance
        best_angle = 0
        max_variance = 0
        
        for angle in np.arange(-15, 15, 1):
            rotated = self._rotate_image(binary, angle)
            # Horizontal projection
            projection = np.sum(rotated, axis=1)
            variance = np.var(projection)
            
            if variance > max_variance:
                max_variance = variance
                best_angle = angle
        
        return best_angle
    
    def _detect_perspective_distortion(self, gray: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Detect perspective distortion by finding plate corners.
        
        Returns:
            (distortion_level, corner_points)
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Look for quadrilateral (4 corners)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(corners)
            
            # Calculate distortion level based on corner angles
            distortion = self._calculate_perspective_distortion(corners)
            
            return distortion, corners
        
        return 0.0, None
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order: TL, TR, BR, BL.
        """
        # Sort by y-coordinate
        corners = corners[np.argsort(corners[:, 1])]
        
        # Top two points
        top = corners[:2]
        top = top[np.argsort(top[:, 0])]
        
        # Bottom two points
        bottom = corners[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def _calculate_perspective_distortion(self, corners: np.ndarray) -> float:
        """
        Calculate the degree of perspective distortion from corner points.
        Returns a value between 0 (no distortion) and 1 (severe distortion).
        """
        # Calculate side lengths
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        
        # Calculate ratios (should be 1.0 for perfect rectangle)
        width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width) if max(top_width, bottom_width) > 0 else 1
        height_ratio = min(left_height, right_height) / max(left_height, right_height) if max(left_height, right_height) > 0 else 1
        
        # Distortion is how far from perfect rectangle (1.0)
        distortion = 1.0 - (width_ratio * height_ratio)
        
        return min(1.0, max(0.0, distortion))
    
    def correct_perspective(self, plate_image: np.ndarray, 
                           corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply perspective transformation to correct distorted plate.
        
        Args:
            plate_image: Input plate image
            corners: Optional pre-detected corners (TL, TR, BR, BL)
            
        Returns:
            Perspective-corrected plate image
        """
        if not self.config.enable_perspective_correction:
            return plate_image
        
        h, w = plate_image.shape[:2]
        
        # Detect corners if not provided
        if corners is None:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
            _, corners = self._detect_perspective_distortion(gray)
        
        if corners is None:
            # Fall back to simple deskew if corners not found
            return self.deskew_plate(plate_image)
        
        # Define destination points (perfect rectangle)
        # Standard Nigerian plate aspect ratio is approximately 3:1
        dst_width = w
        dst_height = int(w / 3)  # Approximate plate aspect ratio
        
        dst_corners = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)
        
        # Compute homography matrix
        try:
            matrix = cv2.getPerspectiveTransform(corners, dst_corners)
            corrected = cv2.warpPerspective(plate_image, matrix, (dst_width, dst_height),
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE)
            
            logger.debug(f"Applied perspective correction: {w}x{h} -> {dst_width}x{dst_height}")
            return corrected
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}")
            return plate_image
    
    def correct_rotation(self, plate_image: np.ndarray, 
                        angle: Optional[float] = None) -> np.ndarray:
        """
        Correct rotation in plate image.
        
        Args:
            plate_image: Input plate image
            angle: Optional pre-detected rotation angle
            
        Returns:
            Rotation-corrected plate image
        """
        if not self.config.enable_rotation_correction:
            return plate_image
        
        # Detect angle if not provided
        if angle is None:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
            angle = self._detect_rotation_angle(gray)
        
        if abs(angle) < 1.0:  # Skip if nearly horizontal
            return plate_image
        
        if abs(angle) > self.config.max_rotation_angle:
            logger.warning(f"Rotation angle {angle:.1f}° exceeds max {self.config.max_rotation_angle}°")
            return plate_image
        
        return self._rotate_image(plate_image, -angle)  # Negate to correct
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle with proper handling of boundaries.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box size
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        # Apply rotation
        rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def deskew_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Apply deskewing to correct horizontal tilt.
        """
        if not self.config.enable_deskew:
            return plate_image
        
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
        skew = self._detect_skew(gray)
        
        if abs(skew) < 1.0:
            return plate_image
        
        return self._rotate_image(plate_image, -skew)
    
    def full_correction(self, plate_image: np.ndarray) -> Tuple[np.ndarray, PlateGeometry]:
        """
        Apply full geometric correction pipeline:
        1. Analyze geometry
        2. Correct perspective
        3. Correct rotation
        4. Apply deskewing
        
        Args:
            plate_image: Input plate image
            
        Returns:
            (corrected_image, geometry_info)
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image, PlateGeometry(0, 0, 0, is_correctable=False)
        
        # Analyze geometry
        geometry = self.analyze_plate_geometry(plate_image)
        
        if not geometry.is_correctable:
            return plate_image, geometry
        
        corrected = plate_image.copy()
        
        # Step 1: Perspective correction (if significant distortion)
        if geometry.perspective_distortion > self.config.perspective_threshold:
            corrected = self.correct_perspective(corrected, geometry.corners)
            logger.debug(f"Applied perspective correction (distortion: {geometry.perspective_distortion:.2f})")
        
        # Step 2: Rotation correction
        if abs(geometry.angle) > 1.0:
            corrected = self.correct_rotation(corrected, geometry.angle)
            logger.debug(f"Applied rotation correction ({geometry.angle:.1f}°)")
        
        # Step 3: Deskewing
        if abs(geometry.skew) > 1.0:
            corrected = self.deskew_plate(corrected)
            logger.debug(f"Applied deskewing ({geometry.skew:.1f}°)")
        
        return corrected, geometry
    
    def generate_multi_angle_variants(self, plate_image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Generate multiple rotated variants of the plate image for multi-angle OCR.
        
        Args:
            plate_image: Input plate image
            
        Returns:
            List of (rotated_image, angle) tuples
        """
        if not self.config.use_multi_angle_ocr:
            return [(plate_image, 0.0)]
        
        variants = []
        for angle in self.config.rotation_angles:
            if angle == 0:
                variants.append((plate_image, 0.0))
            else:
                rotated = self._rotate_image(plate_image, angle)
                variants.append((rotated, angle))
        
        return variants


def preprocess_for_ocr(plate_image: np.ndarray, 
                      apply_geometry_correction: bool = True) -> np.ndarray:
    """
    Comprehensive preprocessing pipeline for OCR on slanted plates.
    
    Args:
        plate_image: Cropped plate image
        apply_geometry_correction: Whether to apply full geometry correction
        
    Returns:
        Preprocessed plate image ready for OCR
    """
    if plate_image is None or plate_image.size == 0:
        return plate_image
    
    processed = plate_image.copy()
    
    # Step 1: Geometry correction
    if apply_geometry_correction:
        corrector = PlateGeometryCorrector()
        processed, _ = corrector.full_correction(processed)
    
    # Step 2: Convert to grayscale for text processing
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed.copy()
    
    # Step 3: Adaptive thresholding for better text contrast
    # This helps with uneven lighting and shadows from angles
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Step 4: Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    
    # Step 5: Resize to optimal OCR height (if needed)
    h, w = cleaned.shape[:2]
    if h < 50:
        scale = 50 / h
        cleaned = cv2.resize(cleaned, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
    
    # Convert back to BGR for consistency
    processed = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    return processed


def enhance_angled_plate(plate_image: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing specifically for angled/slanted plates.
    
    Applies:
    - Bilateral filtering (edge-preserving smoothing)
    - Contrast enhancement
    - Sharpening
    """
    if plate_image is None or plate_image.size == 0:
        return plate_image
    
    processed = plate_image.copy()
    
    # Bilateral filter: smooths while preserving edges (important for text)
    processed = cv2.bilateralFilter(processed, 9, 75, 75)
    
    # Convert to LAB for contrast enhancement
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sharpening
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    processed = cv2.filter2D(processed, -1, kernel)
    
    return processed


# Global corrector instance
_corrector: Optional[PlateGeometryCorrector] = None


def get_geometry_corrector(config: Optional[CorrectionConfig] = None) -> PlateGeometryCorrector:
    """Get or create global geometry corrector."""
    global _corrector
    if _corrector is None:
        _corrector = PlateGeometryCorrector(config)
    return _corrector
