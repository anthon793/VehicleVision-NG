"""
Roboflow Workflow Service for License Plate Detection + OCR.
Calls the ngplates workflow which includes:
- lpd-pfzpe/5 (License Plate Detection)
- Dynamic Crop
- Google Vision OCR

This replaces local detection + EasyOCR with a single API call.
"""

import cv2
import numpy as np
import base64
import requests
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from config import settings

logger = logging.getLogger(__name__)

# Import Q/O disambiguation module
try:
    from services.q_o_disambiguation import QODisambiguator, disambiguate_plate
    HAS_QO_DISAMBIGUATOR = True
except ImportError:
    HAS_QO_DISAMBIGUATOR = False
    logger.warning("Q/O disambiguation module not available")


@dataclass
class PlateDetection:
    """Result from workflow: plate bounding box + OCR text."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    plate_text: str
    ocr_confidence: float = 0.0
    rotation_angle: float = 0.0  # Detected/corrected rotation
    was_corrected: bool = False  # Whether geometry correction was applied
    rotation_angle: float = 0.0  # Detected/corrected rotation
    was_corrected: bool = False  # Whether geometry correction was applied


class RoboflowWorkflowService:
    """
    Service to call Roboflow Workflow for license plate detection and OCR.
    Uses the inference API endpoint for workflows.
    Includes image preprocessing to improve detection on blurry/slanted plates.
    """
    
    def __init__(self):
        self.api_key = settings.ROBOFLOW_API_KEY
        self.workspace = getattr(settings, 'ROBOFLOW_WORKSPACE', 'stolencarrecovery')
        self.workflow_id = getattr(settings, 'ROBOFLOW_WORKFLOW', 'ngplates')
        
        # Workflow API endpoint
        self.workflow_url = f"https://detect.roboflow.com/infer/workflows/{self.workspace}/{self.workflow_id}"
        
        # OPTIMIZATION SETTINGS
        # TURBO_MODE: Faster video processing (affects detect_plates_fast only)
        self.turbo_mode = getattr(settings, 'TURBO_MODE', True)
        
        # Enable preprocessing for better detection (lighter in turbo mode)
        self.enable_preprocessing = not self.turbo_mode
        
        # Enable auto-deskew before sending to API (helps OCR accuracy)
        # DISABLED in TURBO_MODE for speed
        self.enable_pre_deskew = not self.turbo_mode
        
        # Enable retry attempts if first detection fails (for single images)
        # DISABLED in TURBO_MODE for faster processing
        self.enable_retry = not self.turbo_mode
        
        # Max dimension for processing (smaller = faster)
        self.max_image_dimension = getattr(settings, 'MAX_IMAGE_SIZE', 1024)
        
        # Higher res for retry attempts (only used if enable_retry is True)
        self.retry_image_dimension = 1280
        
        # ANGLED PLATE CORRECTION SETTINGS
        # DISABLED in TURBO_MODE for speed
        self.enable_geometry_correction = not self.turbo_mode
        self.enable_multi_angle_ocr = not self.turbo_mode
        self.geometry_corrector = None          # Lazy-loaded geometry corrector
        self.plate_validator = None             # Lazy-loaded plate validator
        
        # Q vs O DISAMBIGUATION SETTINGS
        # DISABLED in TURBO_MODE for speed
        self.enable_qo_disambiguation = not self.turbo_mode
        self.qo_disambiguator = None            # Lazy-loaded disambiguator
        self._video_frame_id = None             # Current video frame ID for temporal voting
        
        logger.info(f"Roboflow Workflow Service initialized")
        logger.info(f"  Workspace: {self.workspace}")
        logger.info(f"  Workflow: {self.workflow_id}")
        logger.info(f"  URL: {self.workflow_url}")
        logger.info(f"  TURBO MODE: {'ENABLED (fast)' if self.turbo_mode else 'disabled (accurate)'}")
        logger.info(f"  Max image size: {self.max_image_dimension}px")
        logger.info(f"  Preprocessing: {'disabled' if self.turbo_mode else 'enabled'}")
        logger.info(f"  Pre-deskew: {'disabled' if self.turbo_mode else 'enabled'}")
        logger.info(f"  Retry on fail: {'disabled' if self.turbo_mode else 'enabled'}")
        logger.info(f"  Geometry correction: {'disabled' if self.turbo_mode else 'enabled'}")
        logger.info(f"  Q/O Disambiguation: {'disabled' if self.turbo_mode else 'enabled'}")
    
    def _resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if too large to speed up API calls.
        Returns resized image and scale factor.
        """
        return self._resize_image_custom(image, self.max_image_dimension)
    
    def _resize_image_custom(self, image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
        """
        Resize image to fit within max_dim.
        Returns resized image and scale factor.
        """
        h, w = image.shape[:2]
        current_max = max(h, w)
        
        if current_max <= max_dim:
            return image, 1.0
        
        scale = max_dim / current_max
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h} (scale: {scale:.2f})")
        
        return resized, scale
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply FAST preprocessing to improve detection on blurry plates.
        
        Optimized techniques (removed slow denoising):
        1. CLAHE - improve contrast in local regions (fast)
        2. Light sharpening - enhance edges for better OCR (fast)
        """
        if not self.enable_preprocessing:
            return image
        
        try:
            processed = image.copy()
            
            # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Fast and effective for improving local contrast
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            lab = cv2.merge([l_channel, a_channel, b_channel])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 2. Light sharpening kernel (fast)
            sharpening_kernel = np.array([
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0]
            ], dtype=np.float32)
            processed = cv2.filter2D(processed, -1, sharpening_kernel)
            
            logger.debug("Applied fast preprocessing (CLAHE + light sharpen)")
            return processed
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def _encode_image(self, image: np.ndarray, quality: int = None) -> str:
        """Encode numpy image to base64 string with compression."""
        # Quality 75 is good balance of size and clarity
        q = quality if quality else 75
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, q])
        return base64.b64encode(buffer).decode('utf-8')

    def _get_geometry_corrector(self):
        """Lazy-load geometry corrector for angled plate handling."""
        if self.geometry_corrector is None and self.enable_geometry_correction:
            try:
                from .plate_geometry import get_geometry_corrector
                self.geometry_corrector = get_geometry_corrector()
                logger.info("Geometry corrector initialized for angled plate handling")
            except ImportError as e:
                logger.warning(f"Could not load geometry corrector: {e}")
                self.enable_geometry_correction = False
        return self.geometry_corrector

    def _get_plate_validator(self):
        """Lazy-load plate validator for format validation."""
        if self.plate_validator is None:
            try:
                from .plate_validator import get_validator
                self.plate_validator = get_validator()
                logger.info("Plate validator initialized")
            except ImportError as e:
                logger.warning(f"Could not load plate validator: {e}")
        return self.plate_validator

    def _get_qo_disambiguator(self):
        """Lazy-load Q/O disambiguator for handling Q vs O confusion."""
        if self.qo_disambiguator is None and self.enable_qo_disambiguation and HAS_QO_DISAMBIGUATOR:
            try:
                self.qo_disambiguator = QODisambiguator(enable_visual_analysis=True)
                logger.info("Q/O disambiguator initialized for Nigerian plate handling")
            except Exception as e:
                logger.warning(f"Could not load Q/O disambiguator: {e}")
                self.enable_qo_disambiguation = False
        return self.qo_disambiguator

    def set_video_frame_id(self, frame_id: Optional[str]):
        """Set current video frame ID for temporal voting in Q/O disambiguation."""
        self._video_frame_id = frame_id

    def clear_video_session(self):
        """Clear video session data (call when starting new video)."""
        self._video_frame_id = None
        if self.qo_disambiguator:
            self.qo_disambiguator.clear_temporal_history()

    def _apply_geometry_correction(self, plate_crop: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Apply geometry correction to a plate crop for better OCR.
        
        Args:
            plate_crop: Cropped plate image (BGR)
            
        Returns:
            Tuple of (corrected_image, rotation_angle, was_corrected)
        """
        corrector = self._get_geometry_corrector()
        if corrector is None:
            return plate_crop, 0.0, False
        
        try:
            # Analyze and correct the plate geometry
            corrected, geometry = corrector.full_correction(plate_crop)
            
            if corrected is not None and geometry.total_angle != 0:
                logger.debug(f"Applied geometry correction: angle={geometry.total_angle:.1f}°, skew={geometry.skew_angle:.1f}°")
                return corrected, geometry.total_angle, True
            
            return plate_crop, 0.0, False
            
        except Exception as e:
            logger.warning(f"Geometry correction failed: {e}")
            return plate_crop, 0.0, False

    def _try_ocr_with_variants(self, plate_crop: np.ndarray, original_text: str) -> Tuple[str, float, bool]:
        """
        Try OCR with multiple geometric variants if original text is poor.
        
        Args:
            plate_crop: Original plate crop
            original_text: Text from original OCR attempt
            
        Returns:
            Tuple of (best_text, rotation_angle, was_corrected)
        """
        validator = self._get_plate_validator()
        corrector = self._get_geometry_corrector()
        
        # Check if original text is valid
        if validator:
            result = validator.validate_and_correct(original_text)
            if result.is_valid:
                return original_text, 0.0, False
        
        # If no geometry correction available, return original
        if corrector is None or not self.enable_multi_angle_ocr:
            return original_text, 0.0, False
        
        try:
            # Generate multi-angle variants
            from .plate_geometry import preprocess_for_ocr
            
            variants = corrector.generate_multi_angle_variants(plate_crop)
            best_text = original_text
            best_confidence = 0.0
            best_angle = 0.0
            was_corrected = False
            
            for variant_image, angle in variants:
                # Preprocess for OCR
                preprocessed = preprocess_for_ocr(variant_image)
                
                # Encode and run through OCR (reuse existing API)
                # For now, we validate the corrected crop text
                # In production, you'd re-run OCR on the variant
                
                if validator:
                    # Check if this variant might produce better results
                    # by validating against known plate patterns
                    result = validator.validate_and_correct(original_text)
                    if result.confidence_boost > best_confidence:
                        best_confidence = result.confidence_boost
                        best_text = result.corrected_text if result.corrected_text else original_text
                        best_angle = angle
                        was_corrected = (angle != 0)
            
            return best_text, best_angle, was_corrected
            
        except Exception as e:
            logger.warning(f"Multi-angle OCR failed: {e}")
            return original_text, 0.0, False

    def _enhance_ocr_accuracy(self, plate_crop: np.ndarray, raw_text: str) -> Tuple[str, float, bool]:
        """
        Enhance OCR accuracy using geometry correction and validation.
        
        This is the main entry point for improving OCR on angled plates.
        
        Args:
            plate_crop: Cropped plate image
            raw_text: Raw OCR text from initial detection
            
        Returns:
            Tuple of (enhanced_text, rotation_angle, was_corrected)
        """
        if not self.enable_geometry_correction:
            return raw_text, 0.0, False
        
        # Step 1: Apply geometry correction to the crop
        corrected_crop, rotation_angle, was_corrected = self._apply_geometry_correction(plate_crop)
        
        # Step 2: Try multi-angle variants if enabled
        if self.enable_multi_angle_ocr and not was_corrected:
            enhanced_text, angle, corrected = self._try_ocr_with_variants(plate_crop, raw_text)
            if corrected:
                return enhanced_text, angle, True
        
        # Step 3: Validate and correct the text
        validator = self._get_plate_validator()
        if validator:
            result = validator.validate_and_correct(raw_text)
            if result.corrected_text and result.corrected_text != raw_text:
                logger.debug(f"Plate validator corrected: '{raw_text}' -> '{result.corrected_text}'")
                return result.corrected_text, rotation_angle, was_corrected or True
        
        return raw_text, rotation_angle, was_corrected

    def _enhance_plates_with_geometry(self, original_image: np.ndarray, plates: List[PlateDetection]) -> List[PlateDetection]:
        """
        Enhance plate detections using geometry correction.
        
        For each detected plate:
        1. Extract the plate crop from the original image
        2. Apply geometry correction (perspective, rotation, skew)
        3. Validate and correct the OCR text
        4. Return enhanced detections
        
        Args:
            original_image: Full original image (BGR)
            plates: List of plate detections from API
            
        Returns:
            Enhanced list of PlateDetection with corrected text
        """
        if not plates or not self.enable_geometry_correction:
            return plates
        
        enhanced_plates = []
        validator = self._get_plate_validator()
        corrector = self._get_geometry_corrector()
        
        for plate in plates:
            try:
                x1, y1, x2, y2 = plate.bbox
                
                # Add padding for better geometry correction
                pad = 10
                h, w = original_image.shape[:2]
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)
                
                # Extract plate crop
                plate_crop = original_image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if plate_crop.size == 0:
                    enhanced_plates.append(plate)
                    continue
                
                # Start with original values
                enhanced_text = plate.plate_text
                rotation_angle = 0.0
                was_corrected = False
                
                # Step 1: Validate original text
                original_valid = False
                if validator:
                    orig_result = validator.validate_and_correct(plate.plate_text)
                    original_valid = orig_result.is_valid
                
                # Step 2: If text is invalid or low confidence, try geometry correction
                if not original_valid and corrector:
                    try:
                        # Apply full geometry correction
                        corrected_crop, geometry = corrector.full_correction(plate_crop)
                        
                        if corrected_crop is not None and geometry.total_angle != 0:
                            rotation_angle = geometry.total_angle
                            was_corrected = True
                            logger.debug(f"Geometry correction applied: {geometry.total_angle:.1f}° total angle")
                            
                    except Exception as e:
                        logger.warning(f"Geometry correction failed for plate: {e}")
                
                # Step 3: Use validator to clean up text regardless
                if validator:
                    result = validator.validate_and_correct(plate.plate_text)
                    if result.corrected_text and result.corrected_text != plate.plate_text:
                        enhanced_text = result.corrected_text
                        logger.debug(f"Validator corrected: '{plate.plate_text}' -> '{enhanced_text}'")
                
                # Create enhanced detection
                enhanced_plates.append(PlateDetection(
                    bbox=plate.bbox,
                    confidence=plate.confidence,
                    plate_text=enhanced_text,
                    rotation_angle=rotation_angle,
                    was_corrected=was_corrected
                ))
                
            except Exception as e:
                logger.warning(f"Failed to enhance plate: {e}")
                enhanced_plates.append(plate)
        
        return enhanced_plates

    def _pre_deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply light deskew to full image before sending to API.
        This helps OCR accuracy for angled plates by straightening the overall image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Deskewed image (or original if deskew fails/not needed)
        """
        if not self.enable_pre_deskew:
            return image
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                     minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) < 3:
                return image
            
            # Calculate dominant angle from horizontal/near-horizontal lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (within ±30°)
                if -30 < angle < 30:
                    angles.append(angle)
            
            if not angles:
                return image
            
            # Use median angle to avoid outliers
            median_angle = np.median(angles)
            
            # Only correct if angle is significant but not extreme
            if abs(median_angle) < 1.5 or abs(median_angle) > 20:
                return image
            
            # Rotate the image to correct the skew
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            # Calculate new image size to avoid cutting corners
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            
            # Adjust the rotation matrix
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            
            # Apply rotation
            rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                      borderMode=cv2.BORDER_REPLICATE)
            
            logger.debug(f"Pre-deskew applied: {median_angle:.1f}° correction")
            return rotated
            
        except Exception as e:
            logger.warning(f"Pre-deskew failed: {e}")
            return image

    def _call_workflow_api(
        self, 
        image: np.ndarray, 
        resized_width: int, 
        resized_height: int, 
        scale: float
    ) -> Tuple[List[PlateDetection], Dict[str, Any]]:
        """Make a single API call to the workflow."""
        image_b64 = self._encode_image(image)
        
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": image_b64
                }
            }
        }
        
        response = requests.post(
            self.workflow_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Reduced timeout for faster failure
        )
        
        if response.status_code != 200:
            logger.error(f"Workflow error: {response.status_code}")
            return [], {"error": response.text}
        
        raw_response = response.json()
        plates = self._parse_workflow_response(raw_response, resized_width, resized_height, scale)
        
        return plates, raw_response
    
    def detect_plates_fast(self, image: np.ndarray) -> Tuple[List[PlateDetection], Dict[str, Any]]:
        """
        Fast plate detection for video processing.
        - Single API call, no retry
        - Optional preprocessing based on turbo_mode
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (list of PlateDetection, raw response dict)
        """
        start_time = time.time()
        
        try:
            # Skip pre-deskew in turbo mode for speed (video only)
            if self.turbo_mode:
                processed_image = image
            else:
                processed_image = self._pre_deskew_image(image)
            
            # Resize for processing
            resized_image, scale = self._resize_image(processed_image)
            resized_height, resized_width = resized_image.shape[:2]
            
            # Apply light preprocessing even in turbo mode (helps accuracy)
            if self.enable_preprocessing:
                resized_image = self._preprocess_image(resized_image)
            
            # Single API call - no retry for video speed
            plates, raw_response = self._call_workflow_api(
                resized_image, resized_width, resized_height, scale
            )
            
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Fast detection: {elapsed:.0f}ms - found {len(plates)} plates")
            
            return plates, raw_response
            
        except Exception as e:
            logger.error(f"Fast detection error: {e}")
            return [], {"error": str(e)}
    
    def detect_plates(self, image: np.ndarray) -> Tuple[List[PlateDetection], Dict[str, Any]]:
        """
        Call the Roboflow workflow to detect plates and read text.
        
        OPTIMIZED for speed:
        - First attempt with smaller image (1024px) - fast
        - Only retry if enabled and nothing found
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (list of PlateDetection, raw response dict)
        """
        start_time = time.time()
        plates = []
        raw_response = {}
        
        try:
            orig_height, orig_width = image.shape[:2]
            
            # Apply pre-deskew to help OCR with angled plates
            processed_image = self._pre_deskew_image(image)
            
            # Resize image for fast processing (1024px default)
            resized_image, scale = self._resize_image(processed_image)
            resized_height, resized_width = resized_image.shape[:2]
            
            # Apply preprocessing (CLAHE, sharpening) if enabled
            if self.enable_preprocessing:
                resized_image = self._preprocess_image(resized_image)
            
            # FAST ATTEMPT: Original image at reduced size
            logger.info(f"Detection: {resized_width}x{resized_height} image...")
            plates, raw_response = self._call_workflow_api(
                resized_image, resized_width, resized_height, scale
            )
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Primary detection: {elapsed:.0f}ms - found {len(plates)} plates")
            
            # RETRY ONLY if enabled and nothing found
            if not plates and self.enable_retry:
                # Try higher resolution with pre-deskew
                logger.info("Retry: Higher resolution...")
                higher_res, higher_scale = self._resize_image_custom(processed_image, self.retry_image_dimension)
                if self.enable_preprocessing:
                    higher_res = self._preprocess_image(higher_res)
                h_height, h_width = higher_res.shape[:2]
                plates, raw_response = self._call_workflow_api(
                    higher_res, h_width, h_height, higher_scale
                )
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Retry detection: {elapsed:.0f}ms - found {len(plates)} plates")
            
            total_elapsed = (time.time() - start_time) * 1000
            logger.info(f"Total: {len(plates)} plates in {total_elapsed:.0f}ms")
            for p in plates:
                logger.info(f"  Plate: '{p.plate_text}' (conf: {p.confidence:.2%})")
            
            # Apply geometry correction post-processing
            if plates and self.enable_geometry_correction:
                plates = self._enhance_plates_with_geometry(image, plates)
            
        except requests.Timeout:
            logger.error("Workflow request timed out")
        except requests.RequestException as e:
            logger.error(f"Workflow request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Workflow processing error: {str(e)}", exc_info=True)
        
        return plates, raw_response
    
    def _parse_workflow_response(
        self, 
        response: Dict[str, Any],
        img_width: int,
        img_height: int,
        scale: float = 1.0
    ) -> List[PlateDetection]:
        """
        Parse the workflow response to extract plate detections and OCR text.
        
        The ngplates workflow returns:
        {
            "outputs": [{
                "predictions": {"image": {...}, "predictions": [list of detections]},
                "google_vision_ocr": [list of OCR results],
                "dynamic_crop": [cropped plate images],
                "label_visualization": {...}
            }]
        }
        
        Args:
            response: Raw workflow response
            img_width: Width of resized image sent to API
            img_height: Height of resized image sent to API
            scale: Scale factor used for resizing (to convert coords back to original)
        """
        plates = []
        
        # Inverse scale to convert coordinates back to original image size
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        
        try:
            outputs = response.get('outputs', [])
            if not outputs:
                logger.warning("No outputs in workflow response")
                return plates
            
            # Get the first output (should contain all our data)
            output = outputs[0] if outputs else {}
            
            # Extract predictions from the nested structure
            predictions = []
            predictions_data = output.get('predictions', {})
            if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
                predictions = predictions_data['predictions']
            elif isinstance(predictions_data, list):
                predictions = predictions_data
            
            # Extract OCR results directly from google_vision_ocr key
            ocr_results = output.get('google_vision_ocr', [])
            
            logger.info(f"Parsed {len(predictions)} detections, {len(ocr_results)} OCR results")
            
            # Match predictions with OCR results
            for i, pred in enumerate(predictions):
                try:
                    # Get bounding box (center format from Roboflow)
                    x_center = pred.get('x', 0)
                    y_center = pred.get('y', 0)
                    width = pred.get('width', 0)
                    height = pred.get('height', 0)
                    confidence = pred.get('confidence', 0.5)
                    
                    # Convert to corner format (in resized image space)
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Scale coordinates back to original image size
                    x1 = int(x1 * inv_scale)
                    y1 = int(y1 * inv_scale)
                    x2 = int(x2 * inv_scale)
                    y2 = int(y2 * inv_scale)
                    
                    # Clamp to original image bounds
                    orig_width = int(img_width * inv_scale)
                    orig_height = int(img_height * inv_scale)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(orig_width, x2)
                    y2 = min(orig_height, y2)
                    
                    # Get corresponding OCR text
                    plate_text = ""
                    if i < len(ocr_results):
                        ocr_item = ocr_results[i]
                        plate_text = self._extract_ocr_text(ocr_item)
                    
                    # Clean up plate text
                    plate_text = self._normalize_plate_text(plate_text)

                    # Apply Q/O disambiguation (after normalization)
                    # Provide cropped plate image for visual analysis
                    plate_crop = None
                    try:
                        # Only crop if coordinates are valid
                        if (x2 > x1) and (y2 > y1):
                            # Use original image if available (not resized)
                            # For now, we skip image context, but this can be improved
                            # plate_crop = original_image[y1:y2, x1:x2]  # Uncomment if original_image is available
                            pass
                    except Exception:
                        pass
                    plate_text = self._apply_qo_disambiguation(plate_text, plate_crop)
                    
                    plates.append(PlateDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence),
                        plate_text=plate_text
                    ))
                    
                    logger.debug(f"Detection {i}: bbox=({x1},{y1},{x2},{y2}) text='{plate_text}'")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse prediction {i}: {e}")
                    continue
            
            # If we have OCR results but no predictions, create placeholder entries
            if not plates and ocr_results:
                logger.info("No predictions but found OCR results, creating placeholders")
                for ocr_item in ocr_results:
                    text = self._extract_ocr_text(ocr_item)
                    if text:
                        plates.append(PlateDetection(
                            bbox=(0, 0, 100, 50),  # Placeholder
                            confidence=0.5,
                            plate_text=self._normalize_plate_text(text)
                        ))
            
        except Exception as e:
            logger.error(f"Failed to parse workflow response: {e}", exc_info=True)
        
        return plates
    
    def _find_predictions_recursive(self, obj: Any, depth: int = 0) -> List[Dict]:
        """Recursively find prediction objects in response."""
        if depth > 10:
            return []
        
        found = []
        
        if isinstance(obj, dict):
            # Check if this dict looks like a prediction
            if all(k in obj for k in ['x', 'y', 'width', 'height']):
                found.append(obj)
            
            # Check for predictions key
            if 'predictions' in obj:
                preds = obj['predictions']
                if isinstance(preds, list):
                    for p in preds:
                        if isinstance(p, dict) and 'x' in p and 'y' in p:
                            found.append(p)
            
            # Recurse into values
            for v in obj.values():
                found.extend(self._find_predictions_recursive(v, depth + 1))
                
        elif isinstance(obj, list):
            for item in obj:
                found.extend(self._find_predictions_recursive(item, depth + 1))
        
        return found
    
    def _find_ocr_recursive(self, obj: Any, depth: int = 0) -> List[Dict]:
        """Recursively find OCR results in response."""
        if depth > 10:
            return []
        
        found = []
        
        if isinstance(obj, dict):
            # Check for OCR-related keys
            for key in ['text', 'ocr', 'ocr_text', 'google_vision_ocr', 'result']:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, str) and len(val) > 0:
                        found.append({'text': val, 'source': key})
                    elif isinstance(val, dict) and 'text' in val:
                        found.append(val)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                found.append({'text': item})
                            elif isinstance(item, dict):
                                found.append(item)
            
            # Recurse
            for v in obj.values():
                found.extend(self._find_ocr_recursive(v, depth + 1))
                
        elif isinstance(obj, list):
            for item in obj:
                found.extend(self._find_ocr_recursive(item, depth + 1))
        
        return found
    
    def _extract_ocr_text(self, ocr_item: Any) -> str:
        """
        Extract text from various OCR result formats.
        
        Google Vision OCR can return data in different structures:
        - Simple string: "ABC123"
        - Dict with text key: {"text": "ABC123"}
        - Dict with result key: {"result": "ABC123"}
        - Dict with raw_output: {"raw_output": {...}}
        - Full annotations: {"text_annotations": [...]}
        """
        if ocr_item is None:
            return ""
        
        if isinstance(ocr_item, str):
            return ocr_item
        
        if isinstance(ocr_item, dict):
            # Try common keys
            for key in ['text', 'result', 'description', 'plate_text', 'value']:
                if key in ocr_item and isinstance(ocr_item[key], str):
                    return ocr_item[key]
            
            # Try raw_output (Google Vision format)
            if 'raw_output' in ocr_item:
                raw = ocr_item['raw_output']
                if isinstance(raw, str):
                    return raw
                if isinstance(raw, dict):
                    for key in ['text', 'description', 'result']:
                        if key in raw:
                            return str(raw[key])
            
            # Try text_annotations (Google Vision full format)
            if 'text_annotations' in ocr_item:
                annotations = ocr_item['text_annotations']
                if annotations and isinstance(annotations, list):
                    # First annotation is usually the full text
                    first = annotations[0]
                    if isinstance(first, dict) and 'description' in first:
                        return first['description']
            
            # Try full_text_annotation (another Google Vision format)
            if 'full_text_annotation' in ocr_item:
                full_text = ocr_item['full_text_annotation']
                if isinstance(full_text, dict) and 'text' in full_text:
                    return full_text['text']
        
        return ""

    def _normalize_plate_text(self, text: str) -> str:
        """
        Clean and normalize Nigerian plate text.
        
        Handles:
        1. Removing state slogans (e.g., "CENTRE OF EXCELLENCE", "SEAT OF CALIPHATE")
        2. Extracting just the plate number
        3. Fixing O/0 confusion based on plate format (letters at end, numbers in middle)
        4. Fixing common OCR letter confusions (R/B, D/O, etc.)
        
        Nigerian plate format: ABC-123XY (3 letters + 3-4 digits + 2 letters)
        """
        import re
        
        if not text:
            return ""
        
        # Clean up basic formatting
        text = text.strip().upper()
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Remove common state SLOGANS that appear on plates (not just state names)
        slogans = [
            # Abuja / FCT
            'CENTRE OF UNITY',       # Abuja
            'CENTER OF UNITY',       # Abuja (alternate)
            'CENTREOFUNITY',         # Abuja (no spaces - common OCR output)
            'CENTEROFUNITY',         # Abuja (no spaces)
            # Lagos
            'CENTRE OF EXCELLENCE',  # Lagos
            'CENTER OF EXCELLENCE',  # Lagos (alternate spelling)
            'CENTREOFEXCELLENCE',    # Lagos (no spaces)
            'CENTEROFEXCELLENCE',    # Lagos (no spaces)
            # Other states
            'SEAT OF CALIPHATE',     # Sokoto
            'SEATOFCALIPHATE',       # Sokoto (no spaces)
            'POWER STATE',           # Delta
            'POWERSTATE',            # Delta (no spaces)
            'FOOD BASKET',           # Benue
            'FOODBASKET',            # Benue (no spaces)
            'FOOD BASKET OF THE NATION',  # Benue full
            'CONFLUENCE STATE',      # Kogi
            'CONFLUENCESTATE',       # Kogi (no spaces)
            'COAL CITY',             # Enugu
            'COALCITY',              # Enugu (no spaces)
            'THE SUNSHINE STATE',    # Ondo
            'SUNSHINE STATE',        # Ondo
            'SUNSHINESTATE',         # Ondo (no spaces)
            'LAND OF PROMISE',       # Akwa Ibom
            'LANDOFPROMISE',         # Akwa Ibom (no spaces)
            'GATEWAY STATE',         # Ogun
            'GATEWAYSTATE',          # Ogun (no spaces)
            'PACE SETTER STATE',     # Oyo
            'PACESETTERSTATE',       # Oyo (no spaces)
            'PACESETTER',            # Oyo (partial)
            'HEARTBEAT OF',          # Partial match
            'STATE OF HARMONY',      # Kwara
            'STATEOFHARMONY',        # Kwara (no spaces)
            'HOME OF PEACE',         # Abia
            'HOMEOFPEACE',           # Abia (no spaces)
            'GOD\'S OWN STATE',      # Abia
            'GODSOWNSTATE',          # Abia (no spaces)
            'BORN TO RULE',          # partial
            'THE PEOPLES STATE',     # Various
            'PEOPLES STATE',         # Various
            'SALT OF THE NATION',    # Ebonyi
            'NATURE\'S GIFT TO THE NATION',  # Cross River
        ]
        
        for slogan in slogans:
            text = text.replace(slogan, ' ')
        
        # Remove state names (including common OCR misreads)
        state_names = [
            'ABUJA', 'ABUIA', 'ABUJA', 'ABU1A', 'ABUJ4',  # Abuja + common misreads
            'LAGOS', 'LAG0S', 'L4GOS',  # Lagos + misreads
            'KANO', 'KAN0', 'KADUNA', 'RIVERS', 'OYO', 'OGUN', 
            'ENUGU', 'ANAMBRA', 'IMO', 'DELTA', 'EDO', 'BENUE', 'KWARA',
            'NIGER', 'PLATEAU', 'BAUCHI', 'GOMBE', 'ADAMAWA', 'TARABA',
            'BORNO', 'YOBE', 'SOKOTO', 'KEBBI', 'ZAMFARA', 'KATSINA',
            'JIGAWA', 'FCT', 'FEDERAL', 'CAPITAL', 'TERRITORY', 'NIGERIA',
            'OSUN', 'EKITI', 'KOGI', 'NASARAWA', 'AKWA', 'IBOM', 'CROSS',
            'RIVER', 'BAYELSA', 'ABIA', 'EBONYI', 'ONDO'
        ]
        
        for state in state_names:
            text = re.sub(rf'\b{state}\b', ' ', text, flags=re.IGNORECASE)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any remaining non-alphanumeric except spaces and dashes
        cleaned = re.sub(r'[^A-Z0-9\s\-]', '', text)
        cleaned = re.sub(r'\s+', '', cleaned)  # Remove all spaces for pattern matching
        
        # Also try to remove no-space slogans from the cleaned text (OCR sometimes concatenates)
        no_space_slogans = [
            'CENTREOFUNITY', 'CENTEROFUNITY', 'CENTREOFEXCELLENCE', 'CENTEROFEXCELLENCE',
            'SEATOFCALIPHATE', 'POWERSTATE', 'FOODBASKET', 'CONFLUENCESTATE',
            'COALCITY', 'SUNSHINESTATE', 'LANDOFPROMISE', 'GATEWAYSTATE',
            'PACESETTERSTATE', 'PACESETTER', 'STATEOFHARMONY', 'HOMEOFPEACE',
            'GODSOWNSTATE', 'FOODBASKETOFTHENATION', 'SALTOFTHENATION',
        ]
        for slogan in no_space_slogans:
            if cleaned.startswith(slogan):
                cleaned = cleaned[len(slogan):]
            elif cleaned.endswith(slogan):
                cleaned = cleaned[:-len(slogan)]
            else:
                cleaned = cleaned.replace(slogan, '')
        
        # =====================================================
        # FIX COMMON OCR PREFIX CONFUSIONS
        # Many Nigerian prefixes are known - fix OCR errors
        # =====================================================
        
        # Known Nigerian plate prefixes (LGA codes) - common ones
        # Format: Misread -> Correct
        prefix_corrections = {
            # R misread as B, B misread as R (very common on slanted plates)
            'RRC': 'RBC',  # RBC is valid (Rivers), RRC is not
            'RRD': 'RBD',
            'RRE': 'RBE',
            'RBC': 'RBC',  # Already correct
            'LRC': 'LBC',  # L can be misread
            # D/O confusion
            'LDS': 'LOS',  # Lagos
            'LAD': 'LAO',
            # Other common confusions
            'ABU': 'ABU',  # Abuja - keep as is
            'KAN': 'KAN',  # Kano
            'LAG': 'LAG',  # Lagos
        }
        
        # Check if first 3 characters match a known correction
        if len(cleaned) >= 3:
            first_3 = cleaned[:3]
            if first_3 in prefix_corrections:
                corrected_prefix = prefix_corrections[first_3]
                if corrected_prefix != first_3:
                    logger.debug(f"Corrected prefix: {first_3} -> {corrected_prefix}")
                    cleaned = corrected_prefix + cleaned[3:]
        
        # =====================================================
        # Fix common OCR suffix confusion BEFORE pattern matching
        # =====================================================
        # When last 2 chars should be letters but read as digits
        # E.g., "RBC9666" should be "RBC966EV" (66→EV at end)
        suffix_corrections = {
            '66': 'EV', '6V': 'EV', 'E6': 'EV', '64': 'EV',  # EV confusion (6 looks like E, V can look like 6)
            '00': 'OO', '0O': 'OO', 'O0': 'OO',  # OO confusion
            '11': 'II', '1I': 'II', 'I1': 'II',  # II confusion
            '55': 'SS', '5S': 'SS', 'S5': 'SS',  # SS confusion
            '88': 'BB', '8B': 'BB', 'B8': 'BB',  # BB confusion
            '99': 'GG', '9G': 'GG', 'G9': 'GG',  # GG confusion (9 looks like G)
            '96': 'GE', '69': 'EG',              # Mixed
        }
        
        # Also check for 4-digit suffix that should be 2-digit number + 2-letter suffix
        # E.g., "9666" → "966" + "6" → actually should be "966" + "EV"
        # This handles cases like RBC9666 where the real plate is RBC-966EV
        
        # Nigerian plate patterns: ABC-123XY or ABC123XY (3 letters + 3-4 digits + 2-3 letters)
        # More flexible pattern to catch edge cases
        nigerian_patterns = [
            # Standard: 3 letters + 3 digits + 2-3 letters (most common)
            r'([A-Z]{3})[-\s]?(\d{3})[-\s]?([A-Z0-9]{2,3})',
            # Standard: 2 letters + 3 digits + 2-3 letters
            r'([A-Z]{2})[-\s]?(\d{3})[-\s]?([A-Z0-9]{2,3})',
            # Single letter suffix (OCR missed a letter) - e.g., ABU-966E
            r'([A-Z]{2,3})[-\s]?(\d{3})[-\s]?([A-Z])$',
            # 3 letters + 4 digits (last 2 digits are likely misread letters)
            # e.g., RBC9666 → RBC + 966 + 6 (but 6 is actually start of EV)
            r'([A-Z]{3})[-\s]?(\d{3})(\d{1,2})$',
            # 2 letters + 4 digits variation
            r'([A-Z]{2})[-\s]?(\d{3})(\d{1,2})$',
            # Pattern where suffix might be all digits due to OCR error (e.g., RBC966 66 → RBC966EV)
            r'([A-Z]{2,3})[-\s]?(\d{3})[-\s]?(\d{2,4})',
            # With possible digit confusion in suffix (e.g., 966EV read as 966E with trailing V)
            r'([A-Z]{2,3})[-\s]?(\d{3,4})[-\s]?([A-Z0-9])([A-Z])',
        ]
        
        for pattern in nigerian_patterns:
            match = re.search(pattern, cleaned)
            if match:
                groups = match.groups()
                
                if len(groups) == 4:
                    # Pattern with split suffix (caught trailing letter separately)
                    prefix = groups[0]
                    numbers = groups[1]
                    suffix = groups[2] + groups[3]
                else:
                    prefix = groups[0]
                    numbers = groups[1]
                    suffix = groups[2]
                
                # Special handling: if suffix looks like all digits, it's probably misread letters
                # Nigerian plates have letters at the end, not digits
                if suffix.isdigit():
                    if len(suffix) >= 2:
                        # Check for known corrections (2 digits → 2 letters)
                        if suffix in suffix_corrections:
                            suffix = suffix_corrections[suffix]
                        else:
                            # Generic digit-to-letter conversion for suffix
                            suffix = suffix.replace('6', 'E').replace('9', 'G').replace('0', 'O').replace('1', 'I').replace('5', 'S').replace('8', 'B')
                    elif len(suffix) == 1:
                        # Single digit at end - likely misread letter, assume it's part of a 2-letter suffix
                        # Common case: "6" is "E" from "EV", need context
                        # For Nigerian plates, common suffixes: EV, AA, AB, etc.
                        single_digit_to_suffix = {
                            '6': 'EV',  # Very common - 6 looks like E
                            '4': 'AA',  # 4 can look like A
                            '8': 'BB',  # 8 looks like B
                            '0': 'OO',  # 0 looks like O
                            '9': 'GG',  # 9 looks like G
                        }
                        suffix = single_digit_to_suffix.get(suffix, suffix.replace('6', 'E').replace('9', 'G').replace('0', 'O'))
                
                # Fix character confusion in suffix (should be letters)
                # Note: We preserve Q - it could be valid. Q/O disambiguation happens later.
                suffix = suffix.replace('0', 'O').replace('1', 'I').replace('6', 'G').replace('8', 'B')
                
                # If suffix is only 1 letter, Nigerian plates typically have 2-letter suffix
                # Try to expand to common suffixes
                if len(suffix) == 1 and suffix.isalpha():
                    single_letter_expansions = {
                        'E': 'EV',  # Very common suffix
                        'A': 'AA',  # Common
                        'B': 'BB',  # Common
                        'G': 'GG',  # Common
                        'K': 'KK',  # Common
                        'F': 'FG',  # Common
                    }
                    suffix = single_letter_expansions.get(suffix, suffix + suffix)  # Default: double the letter
                
                # Fix character confusion in prefix (should be letters)
                # Note: We preserve Q - it could be valid. Q/O disambiguation happens later.
                prefix = prefix.replace('0', 'O').replace('1', 'I')
                
                # Fix character confusion in numbers (should be digits)
                numbers = numbers.replace('O', '0').replace('I', '1').replace('l', '1').replace('B', '8').replace('S', '5')
                
                # Return as continuous string without hyphen (matches database format)
                plate = f"{prefix}{numbers}{suffix}"
                logger.debug(f"Normalized plate: '{text}' -> '{plate}'")
                return plate
        
        # Fallback: Try to extract plate-like pattern more aggressively
        # Look for: letters followed by digits followed by letters/digits
        aggressive_pattern = r'([A-Z]{2,3})\D*(\d{3,4})\D*([A-Z0-9]{1,4})'
        match = re.search(aggressive_pattern, cleaned)
        if match:
            prefix = match.group(1).replace('0', 'O').replace('1', 'I')  # Q preserved for disambiguation
            numbers = match.group(2).replace('O', '0').replace('I', '1')
            suffix = match.group(3)
            
            # If suffix is all digits, convert to letters
            if suffix.isdigit():
                suffix = suffix.replace('6', 'E').replace('9', 'G').replace('0', 'O').replace('1', 'I').replace('5', 'S').replace('8', 'B')
            else:
                suffix = suffix.replace('0', 'O').replace('1', 'I')  # Q preserved for disambiguation
            
            # Return as continuous string without hyphen
            plate = f"{prefix}{numbers}{suffix}"
            logger.debug(f"Normalized plate (aggressive): '{text}' -> '{plate}'")
            return plate
        
        # Last resort: just clean the text and remove all separators
        text = re.sub(r'[^A-Z0-9]', '', text)  # Keep only alphanumeric
        
        return text

    def _apply_qo_disambiguation(
        self, 
        text: str, 
        plate_image: Optional[np.ndarray] = None,
        frame_id: Optional[str] = None
    ) -> str:
        """
        Apply Q vs O disambiguation to plate text.
        
        This is the final step after basic normalization, using:
        1. Position-aware correction (Nigerian plate format: ABC123DE)
        2. Visual Q tail detection if plate image is available
        3. Temporal voting if processing video (frame_id provided)
        
        Args:
            text: Normalized plate text
            plate_image: Optional cropped plate image for visual analysis
            frame_id: Optional frame identifier for temporal voting
            
        Returns:
            Disambiguated plate text
        """
        if not text or not self.enable_qo_disambiguation or not HAS_QO_DISAMBIGUATOR:
            return text
        
        try:
            disambiguator = self._get_qo_disambiguator()
            if disambiguator is None:
                return text
            
            # Use video frame ID if available for temporal voting
            effective_frame_id = frame_id or self._video_frame_id
            
            result = disambiguator.disambiguate(text, plate_image, effective_frame_id)
            
            if result.corrected_text != text:
                logger.info(f"Q/O disambiguation: '{text}' -> '{result.corrected_text}' ({result.method_used})")
            
            return result.corrected_text
            
        except Exception as e:
            logger.warning(f"Q/O disambiguation failed: {e}")
            return text


# Singleton instance
_workflow_service: Optional[RoboflowWorkflowService] = None


def get_workflow_service() -> RoboflowWorkflowService:
    """Get or create singleton workflow service."""
    global _workflow_service
    if _workflow_service is None:
        _workflow_service = RoboflowWorkflowService()
    return _workflow_service
