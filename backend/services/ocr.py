"""
OCR Service Module for License Plate Text Extraction.
Uses EasyOCR with preprocessing optimized for Nigerian license plates.
"""

import cv2
import numpy as np
import easyocr
import re
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRService:
    """
    OCR Service for extracting text from license plate images.
    Optimized for Nigerian license plate formats.
    """
    
    # Nigerian plate patterns (examples):
    # ABC 123 XY, LAG 234 ABC, KJA 567 CD
    # Format: 3 letters + 3 digits + 2 letters (with variations)
    NIGERIAN_PLATE_PATTERN = r'^[A-Z]{2,3}[\s\-]?[0-9]{2,3}[\s\-]?[A-Z]{2,3}$'
    
    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize the OCR service with EasyOCR.
        
        Args:
            languages: List of language codes for OCR
        """
        logger.info("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(languages, gpu=False)  # Set gpu=True if CUDA available
        logger.info("EasyOCR reader initialized successfully")
    
    def preprocess_plate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR accuracy.
        
        Args:
            image: Input plate image (BGR format)
            
        Returns:
            np.ndarray: Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize for better OCR (optimal width around 300-400 pixels)
        height, width = gray.shape[:2]
        if width < 200:
            scale = 300 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding for better character separation
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def normalize_plate_text(self, text: str) -> str:
        """
        Normalize extracted plate text for database comparison.
        
        Args:
            text: Raw OCR output text
            
        Returns:
            str: Normalized plate number (uppercase, no spaces/hyphens)
        """
        # Convert to uppercase
        normalized = text.upper()
        
        # Remove spaces, hyphens, and other separators
        normalized = re.sub(r'[\s\-\.\,\_]', '', normalized)
        
        # Remove any non-alphanumeric characters
        normalized = re.sub(r'[^A-Z0-9]', '', normalized)
        
        # Common OCR corrections for Nigerian plates
        corrections = {
            'O': '0',  # O often confused with 0 in digit positions
            'I': '1',  # I often confused with 1
            'S': '5',  # S sometimes read as 5
            'B': '8',  # B sometimes confused with 8
        }
        
        # Apply corrections only to numeric positions
        # Nigerian format: Letters-Digits-Letters
        if len(normalized) >= 7:
            # Attempt to fix middle digits (positions 2-4 or 3-5)
            chars = list(normalized)
            for i, char in enumerate(chars):
                # If in typical digit position (middle of plate)
                if 2 <= i <= 5 and char in corrections:
                    if normalized[:i].isalpha() or i >= 3:
                        chars[i] = corrections[char]
            normalized = ''.join(chars)
        
        return normalized
    
    def validate_plate_format(self, plate_text: str) -> bool:
        """
        Validate if the extracted text matches Nigerian plate format.
        
        Args:
            plate_text: Normalized plate text
            
        Returns:
            bool: True if valid format, False otherwise
        """
        # Nigerian plates typically have 7-9 characters
        if not (6 <= len(plate_text) <= 10):
            return False
        
        # Check against pattern
        if re.match(self.NIGERIAN_PLATE_PATTERN, plate_text):
            return True
        
        # Relaxed validation for MVP - any alphanumeric string of valid length
        if plate_text.isalnum() and len(plate_text) >= 6:
            return True
        
        return False
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from a license plate image.
        
        Args:
            image: Input plate image (BGR format)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence_score)
        """
        try:
            # Preprocess if requested
            if preprocess:
                processed = self.preprocess_plate_image(image)
            else:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Run OCR
            results = self.reader.readtext(processed, detail=1)
            
            if not results:
                # Try without preprocessing
                if preprocess:
                    results = self.reader.readtext(image, detail=1)
            
            if not results:
                logger.warning("No text detected in plate image")
                return "", 0.0
            
            # Combine all detected text segments
            full_text = ""
            total_confidence = 0.0
            
            for (bbox, text, confidence) in results:
                full_text += text
                total_confidence += confidence
            
            avg_confidence = total_confidence / len(results) if results else 0.0
            
            # Normalize the extracted text
            normalized_text = self.normalize_plate_text(full_text)
            
            logger.info(f"OCR Result: '{full_text}' -> '{normalized_text}' (conf: {avg_confidence:.2f})")
            
            return normalized_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR extraction error: {str(e)}")
            return "", 0.0
    
    def extract_with_multiple_preprocessing(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Try multiple preprocessing methods and return the best result.
        
        Args:
            image: Input plate image
            
        Returns:
            Tuple[str, float]: Best (text, confidence) result
        """
        results = []
        
        # Method 1: Standard preprocessing
        text1, conf1 = self.extract_text(image, preprocess=True)
        if text1:
            results.append((text1, conf1))
        
        # Method 2: No preprocessing
        text2, conf2 = self.extract_text(image, preprocess=False)
        if text2:
            results.append((text2, conf2))
        
        # Method 3: Simple threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text3, conf3 = self.extract_text(binary, preprocess=False)
        if text3:
            results.append((text3, conf3))
        
        if not results:
            return "", 0.0
        
        # Return result with highest confidence that passes validation
        valid_results = [(t, c) for t, c in results if self.validate_plate_format(t)]
        if valid_results:
            return max(valid_results, key=lambda x: x[1])
        
        # If no valid results, return highest confidence anyway
        return max(results, key=lambda x: x[1])


# Singleton instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """
    Get or create singleton OCR service instance.
    
    Returns:
        OCRService: The OCR service instance
    """
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
