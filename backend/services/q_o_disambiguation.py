"""
Q vs O Disambiguation Module for Nigerian License Plates.

Academic Justification (Chapter 4):
- Nigerian plates follow the format ABC123DE (3 letters + 3 digits + 2 letters)
- The letter "Q" is frequently misread as "O" due to:
  - Slanted plate angles causing the Q tail to be invisible
  - Low resolution images
  - Lighting conditions obscuring the Q tail
  - Compression artifacts

This module applies:
1. Position-aware correction (letters at positions 1-3 and 7-8)
2. Character-level visual analysis for Q vs O disambiguation
3. Temporal voting across video frames
4. Regex validation with Nigerian plate format constraints

Author: Stolen Vehicle Detection System
"""

import cv2
import numpy as np
import re
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass, field
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class QODisambiguationResult:
    """Result from Q vs O disambiguation analysis."""
    original_text: str
    corrected_text: str
    confidence: float
    corrections_made: List[str] = field(default_factory=list)
    method_used: str = "none"
    visual_analysis_results: Dict[int, str] = field(default_factory=dict)


@dataclass
class CharacterAnalysis:
    """Result of visual character analysis."""
    position: int
    original_char: str
    detected_char: str
    confidence: float
    has_tail: bool = False
    tail_confidence: float = 0.0


class QODisambiguator:
    """
    Disambiguates Q vs O in Nigerian license plate OCR results.
    
    Nigerian plate format: ABC 123 DE
    - Positions 0-2: Letters (state code)
    - Positions 3-5: Digits
    - Positions 6-7: Letters (suffix)
    
    Q can only appear in letter positions (0-2, 6-7).
    O appearing in these positions may be a misread Q.
    """
    
    # Nigerian plate letter positions (0-indexed, after removing spaces/hyphens)
    LETTER_POSITIONS_8CHAR = {0, 1, 2, 6, 7}  # ABC123DE format
    LETTER_POSITIONS_7CHAR = {0, 1, 5, 6}     # XX123AB format
    DIGIT_POSITIONS_8CHAR = {3, 4, 5}
    DIGIT_POSITIONS_7CHAR = {2, 3, 4}
    
    # Valid Nigerian state codes containing Q
    STATES_WITH_Q = ['AKQ', 'EKQ', 'ABQ']  # Theoretical - Q rare in state codes
    
    # Common Nigerian state codes (for validation)
    NIGERIAN_STATE_CODES = [
        'LAG', 'KAN', 'ABJ', 'FCT', 'RIV', 'OYO', 'KAD', 'EDO', 'DEL', 'ENU',
        'ANM', 'IMO', 'OGU', 'OND', 'KWA', 'AKW', 'ABU', 'ADA', 'BEN', 'BAU',
        'BOR', 'CRS', 'EKI', 'GOB', 'JIG', 'KAT', 'KEB', 'KOG', 'NAS', 'NIG',
        'OSU', 'PLA', 'SOK', 'TAR', 'YOB', 'ZAM', 'EBO', 'BAY', 'GOM',
    ]
    
    def __init__(self, enable_visual_analysis: bool = True):
        """
        Initialize the disambiguator.
        
        Args:
            enable_visual_analysis: Whether to enable visual Q tail detection
        """
        self.enable_visual_analysis = enable_visual_analysis
        self.temporal_history: Dict[str, List[str]] = {}  # For video frame voting
        logger.info(f"QO Disambiguator initialized (visual_analysis={enable_visual_analysis})")
    
    def disambiguate(
        self,
        text: str,
        plate_image: Optional[np.ndarray] = None,
        frame_id: Optional[str] = None
    ) -> QODisambiguationResult:
        """
        Main disambiguation method.
        
        Args:
            text: OCR text result
            plate_image: Optional cropped plate image for visual analysis
            frame_id: Optional frame identifier for temporal voting
            
        Returns:
            QODisambiguationResult with corrected text
        """
        if not text:
            return QODisambiguationResult(
                original_text=text,
                corrected_text=text,
                confidence=0.0
            )
        
        # Normalize text
        normalized = text.upper().replace(' ', '').replace('-', '')
        corrections = []
        method_used = "position_aware"
        visual_results = {}
        
        # Step 1: Position-aware correction
        corrected, pos_corrections = self._position_aware_correction(normalized)
        corrections.extend(pos_corrections)
        
        # Step 2: Visual analysis if image provided and O found in letter positions
        if plate_image is not None and self.enable_visual_analysis:
            o_positions = self._find_o_in_letter_positions(corrected)
            if o_positions:
                visual_corrected, vis_corrections, visual_results = self._visual_q_detection(
                    corrected, plate_image, o_positions
                )
                if vis_corrections:
                    corrected = visual_corrected
                    corrections.extend(vis_corrections)
                    method_used = "visual_analysis"
        
        # Step 3: Temporal voting for video
        if frame_id is not None:
            corrected, temp_corrections = self._apply_temporal_voting(
                corrected, frame_id
            )
            if temp_corrections:
                corrections.extend(temp_corrections)
                method_used = "temporal_voting"
        
        # Step 4: Regex validation
        corrected, regex_corrections = self._regex_validation(corrected)
        corrections.extend(regex_corrections)
        
        # Calculate confidence based on corrections made
        confidence = self._calculate_confidence(corrections, method_used)
        
        # Add formatting back
        formatted = self._format_plate(corrected)
        
        result = QODisambiguationResult(
            original_text=text,
            corrected_text=formatted,
            confidence=confidence,
            corrections_made=corrections,
            method_used=method_used,
            visual_analysis_results=visual_results
        )
        
        if corrections:
            logger.info(f"Q/O disambiguation: '{text}' -> '{formatted}' ({method_used})")
            for c in corrections:
                logger.debug(f"  - {c}")
        
        return result
    
    def _position_aware_correction(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply position-aware Q vs O correction based on Nigerian plate format.
        
        Nigerian format: ABC123DE (8 chars) or XX123AB (7 chars)
        - Letters ONLY at positions 0-2 and 6-7 (8-char) or 0-1 and 5-6 (7-char)
        - Digits ONLY at positions 3-5 (8-char) or 2-4 (7-char)
        
        Rules:
        - "O" in letter position could be "O" or misread "Q"
        - "0" (zero) in letter position should become "O" or "Q"
        - "O" or "Q" in digit position should become "0"
        """
        corrections = []
        chars = list(text)
        n = len(text)
        
        if n == 8:
            letter_positions = self.LETTER_POSITIONS_8CHAR
            digit_positions = self.DIGIT_POSITIONS_8CHAR
        elif n == 7:
            letter_positions = self.LETTER_POSITIONS_7CHAR
            digit_positions = self.DIGIT_POSITIONS_7CHAR
        else:
            # Non-standard length - apply heuristic
            return self._heuristic_correction(text)
        
        for i, char in enumerate(chars):
            if i in letter_positions:
                # Letter position
                if char == '0':  # Zero in letter position
                    # Could be O or Q - default to O, visual analysis may correct
                    chars[i] = 'O'
                    corrections.append(f"Pos {i}: '0' -> 'O' (letter position)")
                elif char.isdigit() and char not in '0':
                    # Other digit in letter position - likely OCR error
                    converted = self._digit_to_letter(char)
                    if converted != char:
                        chars[i] = converted
                        corrections.append(f"Pos {i}: '{char}' -> '{converted}' (letter position)")
            
            elif i in digit_positions:
                # Digit position
                if char == 'O':
                    chars[i] = '0'
                    corrections.append(f"Pos {i}: 'O' -> '0' (digit position)")
                elif char == 'Q':
                    chars[i] = '0'
                    corrections.append(f"Pos {i}: 'Q' -> '0' (digit position)")
                elif char.isalpha():
                    converted = self._letter_to_digit(char)
                    if converted != char:
                        chars[i] = converted
                        corrections.append(f"Pos {i}: '{char}' -> '{converted}' (digit position)")
        
        return ''.join(chars), corrections
    
    def _heuristic_correction(self, text: str) -> Tuple[str, List[str]]:
        """Apply heuristic correction for non-standard plate lengths."""
        corrections = []
        chars = list(text)
        
        # Find letter/digit boundaries
        first_digit_idx = -1
        for i, c in enumerate(chars):
            if c.isdigit() or c in 'OQ':
                first_digit_idx = i
                break
        
        if first_digit_idx < 0:
            return text, corrections
        
        # Everything before first digit should be letters
        for i in range(first_digit_idx):
            if chars[i] == '0':
                chars[i] = 'O'
                corrections.append(f"Pos {i}: '0' -> 'O' (prefix letter)")
        
        # Find where digits end (look for letters after digits)
        last_letter_idx = len(chars)
        for i in range(len(chars) - 1, first_digit_idx, -1):
            if chars[i].isalpha() and chars[i] not in 'OQ':
                last_letter_idx = i
                break
        
        # Everything after last_letter_idx should be letters
        for i in range(last_letter_idx, len(chars)):
            if chars[i] == '0':
                chars[i] = 'O'
                corrections.append(f"Pos {i}: '0' -> 'O' (suffix letter)")
        
        # Middle section should be digits
        for i in range(first_digit_idx, last_letter_idx):
            if chars[i] == 'O':
                chars[i] = '0'
                corrections.append(f"Pos {i}: 'O' -> '0' (digit section)")
            elif chars[i] == 'Q':
                chars[i] = '0'
                corrections.append(f"Pos {i}: 'Q' -> '0' (digit section)")
        
        return ''.join(chars), corrections
    
    def _find_o_in_letter_positions(self, text: str) -> List[int]:
        """Find all 'O' characters that are in letter positions."""
        positions = []
        n = len(text)
        
        if n == 8:
            letter_positions = self.LETTER_POSITIONS_8CHAR
        elif n == 7:
            letter_positions = self.LETTER_POSITIONS_7CHAR
        else:
            # For non-standard, check first 3 and last 2
            letter_positions = {0, 1, 2, n-2, n-1} if n > 5 else {0, 1, 2}
        
        for i, char in enumerate(text):
            if char == 'O' and i in letter_positions:
                positions.append(i)
        
        return positions
    
    def _visual_q_detection(
        self,
        text: str,
        plate_image: np.ndarray,
        o_positions: List[int]
    ) -> Tuple[str, List[str], Dict[int, str]]:
        """
        Visually analyze character regions to detect Q tail.
        
        The Q character has a distinctive tail at the bottom-right.
        We crop each character region and look for this tail using edge detection.
        """
        corrections = []
        visual_results = {}
        chars = list(text)
        
        try:
            # Get plate dimensions
            h, w = plate_image.shape[:2]
            n_chars = len(text)
            
            if n_chars == 0 or w == 0:
                return text, corrections, visual_results
            
            # Estimate character width (assuming roughly equal spacing)
            char_width = w / n_chars
            
            for pos in o_positions:
                # Extract character region with some padding
                x_start = max(0, int(pos * char_width - char_width * 0.1))
                x_end = min(w, int((pos + 1) * char_width + char_width * 0.1))
                
                char_region = plate_image[:, x_start:x_end]
                
                # Analyze for Q tail
                has_tail, tail_confidence = self._detect_q_tail(char_region)
                
                visual_results[pos] = f"tail={has_tail}, conf={tail_confidence:.2f}"
                
                if has_tail and tail_confidence > 0.5:
                    chars[pos] = 'Q'
                    corrections.append(f"Pos {pos}: 'O' -> 'Q' (visual Q tail detected, conf={tail_confidence:.2f})")
            
        except Exception as e:
            logger.warning(f"Visual Q detection failed: {e}")
        
        return ''.join(chars), corrections, visual_results
    
    def _detect_q_tail(self, char_image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect the tail of letter Q in a character image.
        
        The Q tail extends from the bottom-right of the circular part.
        We look for:
        1. A circular/oval shape (common to both O and Q)
        2. An extension at the bottom-right corner (Q tail)
        
        Returns:
            (has_tail, confidence)
        """
        try:
            # Convert to grayscale
            if len(char_image.shape) == 3:
                gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = char_image.copy()
            
            h, w = gray.shape
            if h < 10 or w < 10:
                return False, 0.0
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            
            # Apply threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Try inverted
                _, binary_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False, 0.0
            
            # Get largest contour (likely the character)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(main_contour)
            
            # Analyze bottom-right quadrant for Q tail
            # Q tail extends diagonally from bottom-right of the oval
            bottom_right_region = binary[y + int(bh * 0.6):y + bh, x + int(bw * 0.5):x + bw]
            
            if bottom_right_region.size == 0:
                return False, 0.0
            
            # Look for diagonal pixels in bottom-right (Q tail indicator)
            # The tail should have non-zero pixels extending diagonally
            
            # Apply edge detection on the region
            edges = cv2.Canny(bottom_right_region, 30, 100)
            
            # Count edge pixels in the bottom-right corner
            br_h, br_w = edges.shape
            corner_region = edges[int(br_h * 0.5):, int(br_w * 0.5):]
            
            if corner_region.size == 0:
                return False, 0.0
            
            edge_density = np.sum(corner_region > 0) / corner_region.size
            
            # Q typically has edge density of 0.1-0.3 in bottom-right corner
            # O typically has very low edge density there (< 0.05)
            
            has_tail = edge_density > 0.08
            confidence = min(1.0, edge_density * 5)  # Scale to 0-1
            
            return has_tail, confidence
            
        except Exception as e:
            logger.debug(f"Q tail detection error: {e}")
            return False, 0.0
    
    def _apply_temporal_voting(
        self,
        text: str,
        frame_id: str
    ) -> Tuple[str, List[str]]:
        """
        Apply temporal voting across video frames.
        
        For video processing, we track OCR results across frames and use
        majority voting to determine the correct character.
        
        Args:
            text: Current frame's OCR text
            frame_id: Unique identifier for the plate tracking session
            
        Returns:
            (voted_text, corrections)
        """
        corrections = []
        
        # Extract base plate ID (without Q/O variations) for tracking
        base_id = self._get_base_plate_id(text)
        tracking_key = f"{frame_id}_{base_id}"
        
        # Add to history
        if tracking_key not in self.temporal_history:
            self.temporal_history[tracking_key] = []
        
        self.temporal_history[tracking_key].append(text)
        
        # Need at least 3 readings to vote
        history = self.temporal_history[tracking_key]
        if len(history) < 3:
            return text, corrections
        
        # Use last 5 readings for voting
        recent_readings = history[-5:]
        
        # Vote per character position
        n_chars = len(text)
        voted_chars = list(text)
        
        for pos in range(n_chars):
            chars_at_pos = []
            for reading in recent_readings:
                if len(reading) > pos:
                    chars_at_pos.append(reading[pos])
            
            if chars_at_pos:
                # Count occurrences
                counter = Counter(chars_at_pos)
                most_common_char, count = counter.most_common(1)[0]
                
                # Only override if clear majority (>50%)
                if count > len(chars_at_pos) / 2 and voted_chars[pos] != most_common_char:
                    old_char = voted_chars[pos]
                    voted_chars[pos] = most_common_char
                    corrections.append(
                        f"Pos {pos}: '{old_char}' -> '{most_common_char}' "
                        f"(temporal voting: {count}/{len(chars_at_pos)})"
                    )
        
        return ''.join(voted_chars), corrections
    
    def _get_base_plate_id(self, text: str) -> str:
        """Get base plate ID by replacing Q/O with X for tracking purposes."""
        return text.upper().replace('Q', 'X').replace('O', 'X').replace('0', 'X')
    
    def _regex_validation(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply regex validation for Nigerian plate format.
        
        Valid formats:
        - ABC123DE (most common)
        - XX123ABC
        - ABC1234
        """
        corrections = []
        
        # Already in correct format?
        if self._is_valid_nigerian_format(text):
            return text, corrections
        
        # Try common corrections
        chars = list(text)
        n = len(chars)
        
        # Nigerian plates should not start with digits
        if n >= 2 and chars[0].isdigit():
            # First char might be misread - try converting
            converted = self._digit_to_letter(chars[0])
            if converted != chars[0]:
                chars[0] = converted
                corrections.append(f"Pos 0: '{text[0]}' -> '{converted}' (plates start with letters)")
        
        # Check state code validity
        if n >= 3:
            state_code = ''.join(chars[:3])
            if state_code not in self.NIGERIAN_STATE_CODES:
                # Try common fixes for state codes
                fixed_state = self._fix_state_code(state_code)
                if fixed_state != state_code and fixed_state in self.NIGERIAN_STATE_CODES:
                    chars[:3] = list(fixed_state)
                    corrections.append(f"State code: '{state_code}' -> '{fixed_state}'")
        
        return ''.join(chars), corrections
    
    def _is_valid_nigerian_format(self, text: str) -> bool:
        """Check if text matches Nigerian plate format."""
        patterns = [
            r'^[A-Z]{3}\d{3}[A-Z]{2}$',  # ABC123DE
            r'^[A-Z]{2}\d{3}[A-Z]{3}$',  # XX123ABC
            r'^[A-Z]{3}\d{4}$',          # ABC1234
            r'^[A-Z]{2}\d{3}[A-Z]{2}$',  # XX123AB
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def _fix_state_code(self, code: str) -> str:
        """Try to fix common OCR errors in state codes."""
        # Common fixes
        fixes = {
            '1AG': 'LAG',
            'L4G': 'LAG',
            'LA6': 'LAG',
            'K4N': 'KAN',
            'KAM': 'KAN',
            '0YO': 'OYO',
            'OY0': 'OYO',
            'AB1': 'ABJ',
            'A8J': 'ABJ',
        }
        return fixes.get(code, code)
    
    def _digit_to_letter(self, digit: str) -> str:
        """Convert digit to most likely letter."""
        mapping = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '8': 'B',
        }
        return mapping.get(digit, digit)
    
    def _letter_to_digit(self, letter: str) -> str:
        """Convert letter to most likely digit."""
        mapping = {
            'O': '0',
            'Q': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'S': '5',
            'G': '6',
            'B': '8',
        }
        return mapping.get(letter.upper(), letter)
    
    def _calculate_confidence(self, corrections: List[str], method: str) -> float:
        """Calculate confidence based on corrections made."""
        base_confidence = 0.9
        
        # Reduce confidence for each correction
        confidence = base_confidence - (len(corrections) * 0.05)
        
        # Boost for certain methods
        if method == "visual_analysis":
            confidence += 0.05
        elif method == "temporal_voting":
            confidence += 0.1
        
        return max(0.5, min(1.0, confidence))
    
    def _format_plate(self, text: str) -> str:
        """Add standard formatting to plate number."""
        n = len(text)
        
        if n == 8:
            # ABC 123 DE
            return f"{text[:3]}-{text[3:6]}{text[6:]}"
        elif n == 7:
            # XX 123 AB or ABC 1234
            if text[2].isdigit():
                return f"{text[:2]}-{text[2:5]}{text[5:]}"
            else:
                return f"{text[:3]}-{text[3:]}"
        elif n == 6:
            return f"{text[:3]}-{text[3:]}"
        
        return text
    
    def clear_temporal_history(self, frame_id: Optional[str] = None):
        """Clear temporal voting history."""
        if frame_id:
            keys_to_remove = [k for k in self.temporal_history if k.startswith(f"{frame_id}_")]
            for k in keys_to_remove:
                del self.temporal_history[k]
        else:
            self.temporal_history.clear()


# Singleton instance
_disambiguator: Optional[QODisambiguator] = None


def get_disambiguator() -> QODisambiguator:
    """Get or create singleton disambiguator."""
    global _disambiguator
    if _disambiguator is None:
        _disambiguator = QODisambiguator()
    return _disambiguator


def disambiguate_plate(
    text: str,
    plate_image: Optional[np.ndarray] = None,
    frame_id: Optional[str] = None
) -> QODisambiguationResult:
    """
    Convenience function to disambiguate Q vs O in plate text.
    
    Args:
        text: OCR result text
        plate_image: Optional cropped plate image for visual analysis
        frame_id: Optional frame ID for temporal voting (video processing)
        
    Returns:
        QODisambiguationResult with corrected text
    """
    return get_disambiguator().disambiguate(text, plate_image, frame_id)
