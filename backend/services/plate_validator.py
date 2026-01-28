"""
Nigerian License Plate Validation and OCR Post-Processing Module.

Academic Justification (Chapter 4):
- Nigerian plates follow specific formats (ABC 123 DE, LAG 123 ABC, etc.)
- Character confusion is common in OCR (0/O, 1/I, 8/B, 5/S)
- Regex validation and correction improves accuracy significantly
"""

import re
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Nigerian plate format patterns
# Format 1: ABC 123 DE (common format)
# Format 2: LAG 123 ABC (state code prefix)
# Format 3: 123 ABC (older format)
# Format 4: Commercial plates, diplomatic plates, etc.

NIGERIAN_PLATE_PATTERNS = [
    # Standard format: XXX 000 XX or XXX-000-XX
    r'^[A-Z]{3}[\s\-]?\d{3}[\s\-]?[A-Z]{2}$',
    
    # State code format: XX 000 XXX
    r'^[A-Z]{2}[\s\-]?\d{3}[\s\-]?[A-Z]{3}$',
    
    # Older format: 000 XXX
    r'^\d{3}[\s\-]?[A-Z]{3}$',
    
    # Alternative: XXX 0000
    r'^[A-Z]{3}[\s\-]?\d{4}$',
    
    # Commercial/Government: X 000 XXX or XX 000 XX
    r'^[A-Z]{1,2}[\s\-]?\d{3}[\s\-]?[A-Z]{2,3}$',
    
    # Relaxed pattern for partial matches
    r'^[A-Z0-9]{2,4}[\s\-]?[A-Z0-9]{2,4}[\s\-]?[A-Z0-9]{2,4}$',
]

# Character confusion mapping (what OCR might misread)
CHAR_CONFUSION_MAP = {
    # Letters that look like numbers
    'O': ['0', 'Q', 'D'],
    'I': ['1', 'L', '|'],
    'S': ['5', '$'],
    'B': ['8', '3'],
    'Z': ['2', '7'],
    'G': ['6', 'C'],
    'T': ['7', '1'],
    'A': ['4'],
    
    # Numbers that look like letters
    '0': ['O', 'Q', 'D'],
    '1': ['I', 'L', '|', 'T'],
    '2': ['Z'],
    '5': ['S', '$'],
    '8': ['B', '&'],
    '6': ['G', 'b'],
    '4': ['A'],
}

# Nigerian state codes (first 2-3 letters)
NIGERIAN_STATE_CODES = [
    'LAG', 'KAN', 'ABJ', 'FCT', 'RIV', 'OYO', 'KAD', 'EDO', 'DEL', 'ENU',
    'ANM', 'IMO', 'OGU', 'OND', 'KWA', 'AKW', 'ABU', 'ADA', 'BEN', 'BAU',
    'BOR', 'CRS', 'EKI', 'GOB', 'JIG', 'KAT', 'KEB', 'KOG', 'NAS', 'NIG',
    'OSU', 'PLA', 'SOK', 'TAR', 'YOB', 'ZAM', 'EBO', 'BAY', 'GOM',
    # Short codes
    'LA', 'KN', 'AB', 'FC', 'RV', 'OY', 'KD', 'ED', 'DL', 'EN',
]


@dataclass
class PlateValidationResult:
    """Result of plate validation and correction."""
    original_text: str
    corrected_text: str
    is_valid: bool
    confidence_boost: float  # How much to adjust confidence based on validation
    matched_pattern: Optional[str]
    corrections_made: List[str]


class NigerianPlateValidator:
    """
    Validates and corrects Nigerian license plate OCR results.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, only accept plates matching known formats exactly
        """
        self.strict_mode = strict_mode
        self.patterns = [re.compile(p) for p in NIGERIAN_PLATE_PATTERNS]
        logger.info(f"Nigerian Plate Validator initialized (strict={strict_mode})")
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize OCR text for validation.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized text (uppercase, cleaned)
        """
        if not text:
            return ""
        
        # Uppercase
        text = text.upper()
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')
        text = text.replace('$', 'S')
        text = text.replace('&', 'B')
        text = text.replace('#', '')
        text = text.replace('*', '')
        text = text.replace('@', 'A')
        
        # Keep only alphanumeric and spaces/hyphens
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing hyphens
        text = text.strip('-')
        
        return text
    
    def validate_format(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text matches any Nigerian plate format.
        
        Args:
            text: Normalized plate text
            
        Returns:
            (is_valid, matched_pattern_string)
        """
        # Remove spaces and hyphens for pattern matching
        compact = re.sub(r'[\s\-]', '', text)
        
        for i, pattern in enumerate(self.patterns):
            if pattern.match(text) or pattern.match(compact):
                return True, NIGERIAN_PLATE_PATTERNS[i]
        
        return False, None
    
    def correct_character_confusion(
        self,
        text: str,
        position_hints: Optional[Dict[int, str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Correct common OCR character confusions.
        
        Nigerian plates typically have format: [LETTERS][NUMBERS][LETTERS]
        We use position to determine if character should be letter or number.
        
        Args:
            text: Input text
            position_hints: Optional dict mapping positions to 'letter' or 'number'
            
        Returns:
            (corrected_text, list of corrections made)
        """
        if not text:
            return text, []
        
        corrections = []
        compact = re.sub(r'[\s\-]', '', text)
        
        # Infer position hints if not provided
        if position_hints is None:
            position_hints = self._infer_position_hints(compact)
        
        result_chars = []
        for i, char in enumerate(compact):
            expected_type = position_hints.get(i)
            
            if expected_type == 'letter' and char.isdigit():
                # Number in letter position - try to convert
                corrected = self._number_to_letter(char)
                if corrected != char:
                    corrections.append(f"{char}->{corrected} at pos {i}")
                result_chars.append(corrected)
            
            elif expected_type == 'number' and char.isalpha():
                # Letter in number position - try to convert
                corrected = self._letter_to_number(char)
                if corrected != char:
                    corrections.append(f"{char}->{corrected} at pos {i}")
                result_chars.append(corrected)
            
            else:
                result_chars.append(char)
        
        # Reconstruct with standard spacing
        corrected = ''.join(result_chars)
        corrected = self._add_standard_spacing(corrected)
        
        return corrected, corrections
    
    def _infer_position_hints(self, text: str) -> Dict[int, str]:
        """
        Infer whether each position should be letter or number.
        Based on common Nigerian plate formats.
        """
        hints = {}
        n = len(text)
        
        if n == 8:  # ABC123DE format
            for i in range(3):
                hints[i] = 'letter'
            for i in range(3, 6):
                hints[i] = 'number'
            for i in range(6, 8):
                hints[i] = 'letter'
        
        elif n == 7:  # XX123ABC or ABC1234 format
            # Check which format it looks like
            letters_at_start = sum(1 for c in text[:2] if c.isalpha())
            if letters_at_start >= 1:
                # XX 123 ABC format
                hints[0] = 'letter'
                hints[1] = 'letter'
                for i in range(2, 5):
                    hints[i] = 'number'
                for i in range(5, 7):
                    hints[i] = 'letter'
            else:
                # ABC 1234 format
                for i in range(3):
                    hints[i] = 'letter'
                for i in range(3, 7):
                    hints[i] = 'number'
        
        elif n == 6:  # 123ABC format
            for i in range(3):
                hints[i] = 'number'
            for i in range(3, 6):
                hints[i] = 'letter'
        
        return hints
    
    def _number_to_letter(self, num: str) -> str:
        """Convert number to most likely letter."""
        mapping = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '8': 'B',
        }
        return mapping.get(num, num)
    
    def _letter_to_number(self, letter: str) -> str:
        """Convert letter to most likely number."""
        mapping = {
            'O': '0',
            'Q': '0',
            'D': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'A': '4',
            'S': '5',
            'G': '6',
            'B': '8',
            'T': '7',
        }
        return mapping.get(letter.upper(), letter)
    
    def _add_standard_spacing(self, text: str) -> str:
        """Add standard spacing to plate text."""
        n = len(text)
        
        if n == 8:  # ABC 123 DE
            return f"{text[:3]} {text[3:6]} {text[6:]}"
        elif n == 7:
            # Check format
            if text[2].isdigit():
                return f"{text[:2]} {text[2:5]} {text[5:]}"
            else:
                return f"{text[:3]} {text[3:]}"
        elif n == 6:
            return f"{text[:3]} {text[3:]}"
        
        return text
    
    def validate_and_correct(self, text: str) -> PlateValidationResult:
        """
        Full validation and correction pipeline.
        
        Args:
            text: Raw OCR text
            
        Returns:
            PlateValidationResult with corrected text and validation info
        """
        original = text
        corrections = []
        
        # Step 1: Normalize
        normalized = self.normalize_text(text)
        if normalized != text:
            corrections.append(f"Normalized: '{text}' -> '{normalized}'")
        
        # Step 2: Check if already valid
        is_valid, pattern = self.validate_format(normalized)
        
        if is_valid:
            return PlateValidationResult(
                original_text=original,
                corrected_text=normalized,
                is_valid=True,
                confidence_boost=0.1,  # Valid format = confidence boost
                matched_pattern=pattern,
                corrections_made=corrections
            )
        
        # Step 3: Try character corrections
        corrected, char_corrections = self.correct_character_confusion(normalized)
        corrections.extend(char_corrections)
        
        # Step 4: Re-validate after corrections
        is_valid, pattern = self.validate_format(corrected)
        
        confidence_boost = 0.0
        if is_valid:
            confidence_boost = 0.05  # Valid after correction = smaller boost
        elif len(corrected) >= 6 and not self.strict_mode:
            # Partial credit for plate-like strings
            is_valid = True
            confidence_boost = -0.05  # Uncertainty = confidence reduction
        
        return PlateValidationResult(
            original_text=original,
            corrected_text=corrected,
            is_valid=is_valid,
            confidence_boost=confidence_boost,
            matched_pattern=pattern,
            corrections_made=corrections
        )
    
    def check_state_code(self, text: str) -> Optional[str]:
        """
        Extract and validate Nigerian state code from plate.
        
        Args:
            text: Plate text
            
        Returns:
            State code if found, None otherwise
        """
        normalized = self.normalize_text(text)
        compact = normalized.replace(' ', '').replace('-', '')
        
        # Check 3-letter codes first
        for code in NIGERIAN_STATE_CODES:
            if len(code) == 3 and compact.startswith(code):
                return code
        
        # Check 2-letter codes
        for code in NIGERIAN_STATE_CODES:
            if len(code) == 2 and compact.startswith(code):
                return code
        
        return None
    
    def get_plate_type(self, text: str) -> str:
        """
        Determine plate type (private, commercial, government, diplomatic).
        Based on format and prefix patterns.
        """
        normalized = self.normalize_text(text)
        compact = normalized.replace(' ', '').replace('-', '')
        
        # Check for diplomatic prefix
        if compact.startswith('D') or compact.startswith('CD'):
            return 'diplomatic'
        
        # Check for military prefix
        if compact.startswith('NA') or compact.startswith('AF'):
            return 'military'
        
        # Check for government
        if compact.startswith('FG') or compact.startswith('SG'):
            return 'government'
        
        # Default to private
        return 'private'


def process_ocr_result(
    text: str,
    strict: bool = False
) -> PlateValidationResult:
    """
    Convenience function to process OCR result.
    
    Args:
        text: Raw OCR text
        strict: Whether to use strict validation mode
        
    Returns:
        PlateValidationResult
    """
    validator = NigerianPlateValidator(strict_mode=strict)
    return validator.validate_and_correct(text)


def batch_process_ocr_results(
    texts: List[str],
    strict: bool = False
) -> List[PlateValidationResult]:
    """
    Process multiple OCR results with single validator instance.
    """
    validator = NigerianPlateValidator(strict_mode=strict)
    return [validator.validate_and_correct(text) for text in texts]


# Singleton validator
_validator: Optional[NigerianPlateValidator] = None


def get_validator(strict: bool = False) -> NigerianPlateValidator:
    """Get or create singleton validator."""
    global _validator
    if _validator is None:
        _validator = NigerianPlateValidator(strict_mode=strict)
    return _validator
