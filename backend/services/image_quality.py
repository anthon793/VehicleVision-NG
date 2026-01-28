"""
Image Quality Enhancement Module.
Handles blur detection, adaptive preprocessing, deskewing, and super-resolution.

Academic Justification (Chapter 4):
- Nigerian road conditions create motion blur, poor lighting
- Preprocessing improves OCR accuracy on degraded images
- Super-resolution helps with distant/small plates
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Image quality assessment metrics."""
    blur_score: float        # Laplacian variance (higher = sharper)
    brightness: float        # Mean pixel value (0-255)
    contrast: float          # Standard deviation of pixels
    is_blurry: bool          # Below blur threshold
    is_dark: bool            # Below brightness threshold
    is_low_contrast: bool    # Below contrast threshold
    overall_quality: str     # "good", "medium", "poor"


@dataclass  
class QualityConfig:
    """Configuration for quality assessment and enhancement."""
    blur_threshold: float = 100.0       # Laplacian variance threshold
    brightness_low: float = 50.0        # Too dark threshold
    brightness_high: float = 200.0      # Too bright threshold
    contrast_threshold: float = 30.0    # Minimum std dev
    enable_preprocessing: bool = True
    enable_super_resolution: bool = True
    min_plate_size_for_sr: int = 50    # Min dimension to trigger SR
    max_plate_size_for_sr: int = 200   # Max dimension for SR (already large enough)


class ImageQualityProcessor:
    """
    Assesses and enhances image quality for better plate detection.
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize quality processor."""
        self.config = config or QualityConfig()
        self._sr_model = None  # Lazy loaded super-resolution model
        logger.info("Image Quality Processor initialized")
    
    def assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Assess image quality metrics.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            QualityMetrics with assessment results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        # Brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Assessments
        is_blurry = blur_score < self.config.blur_threshold
        is_dark = brightness < self.config.brightness_low
        is_bright = brightness > self.config.brightness_high
        is_low_contrast = contrast < self.config.contrast_threshold
        
        # Overall quality
        issues = sum([is_blurry, is_dark or is_bright, is_low_contrast])
        if issues == 0:
            overall = "good"
        elif issues == 1:
            overall = "medium"
        else:
            overall = "poor"
        
        return QualityMetrics(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            is_blurry=is_blurry,
            is_dark=is_dark or is_bright,
            is_low_contrast=is_low_contrast,
            overall_quality=overall
        )
    
    def enhance_image(
        self, 
        image: np.ndarray,
        quality: Optional[QualityMetrics] = None
    ) -> np.ndarray:
        """
        Apply adaptive enhancement based on quality assessment.
        
        Args:
            image: Input image
            quality: Pre-computed quality metrics (computed if not provided)
            
        Returns:
            Enhanced image
        """
        if not self.config.enable_preprocessing:
            return image
        
        if quality is None:
            quality = self.assess_quality(image)
        
        enhanced = image.copy()
        
        # Handle dark images
        if quality.is_dark:
            enhanced = self._enhance_brightness(enhanced)
            logger.debug("Applied brightness enhancement")
        
        # Handle low contrast
        if quality.is_low_contrast:
            enhanced = self._enhance_contrast(enhanced)
            logger.debug("Applied contrast enhancement")
        
        # Handle blur (limited - can't fully deblur, but can sharpen)
        if quality.is_blurry:
            enhanced = self._apply_sharpening(enhanced)
            logger.debug("Applied sharpening")
        
        return enhanced
    
    def _enhance_brightness(self, image: np.ndarray) -> np.ndarray:
        """Enhance brightness using CLAHE in LAB color space."""
        if len(image.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        
        # Color image - convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using histogram equalization."""
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        
        # For color images, use CLAHE on L channel
        return self._enhance_brightness(image)
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp mask for sharpening."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def deskew_plate(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Correct skew in plate image for better OCR.
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            Deskewed plate image
        """
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            return plate_image
        
        # Calculate dominant angle
        angles = []
        for line in lines[:10]:  # Use top 10 lines
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # Reasonable skew range
                angles.append(angle)
        
        if not angles:
            return plate_image
        
        # Use median angle for robustness
        median_angle = np.median(angles)
        
        if abs(median_angle) < 1.0:  # Don't deskew if nearly level
            return plate_image
        
        # Rotate to correct skew
        h, w = plate_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new bounding box
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        
        deskewed = cv2.warpAffine(
            plate_image, matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug(f"Deskewed plate by {median_angle:.1f} degrees")
        return deskewed
    
    def super_resolve(
        self, 
        image: np.ndarray,
        scale: int = 2
    ) -> np.ndarray:
        """
        Apply super-resolution to upscale small plate images.
        Uses bicubic interpolation with enhancement (no deep learning for simplicity).
        
        For production, consider using ESPCN, FSRCNN, or Real-ESRGAN models.
        
        Args:
            image: Input image
            scale: Upscale factor (2 or 4)
            
        Returns:
            Super-resolved image
        """
        if not self.config.enable_super_resolution:
            return image
        
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # Check if SR is needed
        if min_dim > self.config.max_plate_size_for_sr:
            return image  # Already large enough
        
        if min_dim < self.config.min_plate_size_for_sr:
            logger.warning(f"Image too small for reliable SR: {min_dim}px")
        
        # Upscale using bicubic (fast, acceptable quality)
        new_h, new_w = h * scale, w * scale
        upscaled = cv2.resize(
            image, (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply mild sharpening to the upscaled image
        upscaled = self._apply_sharpening(upscaled)
        
        # Apply CLAHE for better contrast
        upscaled = self._enhance_brightness(upscaled)
        
        logger.debug(f"Super-resolved {w}x{h} -> {new_w}x{new_h}")
        return upscaled
    
    def adaptive_preprocess_for_ocr(
        self,
        plate_image: np.ndarray
    ) -> np.ndarray:
        """
        Apply comprehensive preprocessing optimized for OCR.
        
        Args:
            plate_image: Cropped plate region
            
        Returns:
            Preprocessed plate image ready for OCR
        """
        h, w = plate_image.shape[:2]
        processed = plate_image.copy()
        
        # 1. Super-resolve if too small
        if min(h, w) < 80:
            processed = self.super_resolve(processed, scale=2)
        
        # 2. Quality assessment and enhancement
        quality = self.assess_quality(processed)
        processed = self.enhance_image(processed, quality)
        
        # 3. Deskew
        processed = self.deskew_plate(processed)
        
        # 4. Final resize to optimal OCR size (if needed)
        h, w = processed.shape[:2]
        if w < 200:
            scale = 200 / w
            new_h, new_w = int(h * scale), 200
            processed = cv2.resize(
                processed, (new_w, new_h),
                interpolation=cv2.INTER_CUBIC
            )
        
        return processed


def detect_motion_blur(image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect motion blur in image.
    
    Returns:
        (is_motion_blurred, blur_direction_angle)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # FFT analysis for motion blur detection
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Log magnitude spectrum
    magnitude_log = np.log1p(magnitude)
    
    # Look for directional patterns indicating motion blur
    h, w = magnitude_log.shape
    center_h, center_w = h // 2, w // 2
    
    # Sample horizontal and vertical profiles
    horiz_profile = magnitude_log[center_h, :]
    vert_profile = magnitude_log[:, center_w]
    
    horiz_energy = np.sum(horiz_profile)
    vert_energy = np.sum(vert_profile)
    
    ratio = horiz_energy / (vert_energy + 1e-10)
    
    # Motion blur creates streaks in one direction
    is_motion_blurred = ratio > 1.5 or ratio < 0.67
    blur_angle = 0 if ratio > 1.5 else 90
    
    return is_motion_blurred, blur_angle


def apply_majority_voting(
    ocr_results: List[str],
    char_level: bool = False
) -> str:
    """
    Apply majority voting across multiple OCR results.
    
    Args:
        ocr_results: List of OCR text results
        char_level: If True, vote character-by-character
        
    Returns:
        Most likely correct text
    """
    if not ocr_results:
        return ""
    
    if len(ocr_results) == 1:
        return ocr_results[0]
    
    if not char_level:
        # Simple majority voting on full strings
        from collections import Counter
        counts = Counter(ocr_results)
        return counts.most_common(1)[0][0]
    
    # Character-level voting
    max_len = max(len(r) for r in ocr_results)
    result_chars = []
    
    for i in range(max_len):
        chars_at_pos = []
        for r in ocr_results:
            if i < len(r):
                chars_at_pos.append(r[i])
        
        if chars_at_pos:
            from collections import Counter
            counts = Counter(chars_at_pos)
            result_chars.append(counts.most_common(1)[0][0])
    
    return ''.join(result_chars)


# Global instance
_quality_processor: Optional[ImageQualityProcessor] = None


def get_quality_processor(config: Optional[QualityConfig] = None) -> ImageQualityProcessor:
    """Get or create global quality processor."""
    global _quality_processor
    if _quality_processor is None:
        _quality_processor = ImageQualityProcessor(config)
    return _quality_processor
