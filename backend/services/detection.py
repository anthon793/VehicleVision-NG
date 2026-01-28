"""
Vehicle and License Plate Detection Service.
Uses YOLOv8 models for two-stage detection pipeline.
Supports Roboflow API for enhanced license plate detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import base64
import os
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import roboflow for API-based detection
try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    logger.warning("Roboflow not installed. Install with: pip install roboflow")


class DetectionResult:
    """Data class for detection results."""
    
    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        class_name: str,
        cropped_image: Optional[np.ndarray] = None
    ):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.cropped_image = cropped_image
    
    def to_dict(self) -> Dict:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }


class DetectionService:
    """
    Detection service for vehicles and license plates using YOLOv8.
    Implements a two-stage detection approach.
    """
    
    # COCO vehicle class IDs
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    def __init__(
        self,
        vehicle_model_path: Optional[str] = None,
        plate_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize detection service with YOLOv8 models.
        
        Args:
            vehicle_model_path: Path to vehicle detection model
            plate_model_path: Path to license plate detection model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.vehicle_model = None
        self.plate_model = None
        self.roboflow_plate_model = None
        self.roboflow_workflow = None
        
        # Load vehicle detection model (YOLOv8 pretrained on COCO)
        vehicle_path = vehicle_model_path or settings.VEHICLE_MODEL_PATH
        self._load_vehicle_model(vehicle_path)
        
        # Load plate detection model (custom trained or fallback)
        plate_path = plate_model_path or settings.PLATE_MODEL_PATH
        self._load_plate_model(plate_path)
        
        # Try to initialize Roboflow for better plate detection
        self._init_roboflow_plate_detection()
    
    def _load_vehicle_model(self, model_path: str):
        """Load the vehicle detection model."""
        try:
            # Check if custom model exists, otherwise use pretrained
            if Path(model_path).exists():
                logger.info(f"Loading vehicle model from: {model_path}")
                self.vehicle_model = YOLO(model_path)
            else:
                logger.info("Loading pretrained YOLOv8n model for vehicle detection")
                self.vehicle_model = YOLO("yolov8n.pt")  # Will download if not present
            logger.info("Vehicle detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vehicle model: {str(e)}")
            raise
    
    def _load_plate_model(self, model_path: str):
        """Load the license plate detection model."""
        try:
            if Path(model_path).exists():
                logger.info(f"Loading plate model from: {model_path}")
                self.plate_model = YOLO(model_path)
                logger.info("Plate detection model loaded successfully")
            else:
                logger.warning(f"Plate model not found at: {model_path}")
                logger.info("Will try Roboflow API or use plate region estimation")
                self.plate_model = None
        except Exception as e:
            logger.error(f"Failed to load plate model: {str(e)}")
            self.plate_model = None
    
    def _init_roboflow_plate_detection(self):
        """Initialize Roboflow API for license plate detection."""
        if not ROBOFLOW_AVAILABLE:
            logger.info("Roboflow not available for plate detection")
            return
        
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            logger.warning("ROBOFLOW_API_KEY not set. Get one at https://app.roboflow.com/settings/api")
            logger.info("Using fallback plate detection method")
            return
        
        try:
            rf = Roboflow(api_key=api_key)

            # If workflow details are provided, prefer workflow
            if getattr(settings, "ROBOFLOW_USE_WORKFLOW", False) and getattr(settings, "ROBOFLOW_WORKFLOW", ""):
                try:
                    workspace_slug = getattr(settings, "ROBOFLOW_WORKSPACE", None)
                    workflow_slug = getattr(settings, "ROBOFLOW_WORKFLOW", None)
                    logger.info("Attempting to initialize Roboflow Workflow: %s/%s", workspace_slug, workflow_slug)
                    if workspace_slug:
                        ws = rf.workspace(workspace_slug)
                    else:
                        ws = rf.workspace()
                    # SDK method name may vary; try common options
                    wf = None
                    for attr in ("workflow", "workflows"):
                        if hasattr(ws, attr):
                            try:
                                wf = getattr(ws, attr)(workflow_slug)
                                break
                            except Exception:
                                pass
                    self.roboflow_workflow = wf
                    if self.roboflow_workflow is not None:
                        logger.info("Roboflow workflow initialized: %s", workflow_slug)
                except Exception as wf_err:
                    logger.warning("Failed to initialize Roboflow workflow: %s", str(wf_err))

            # Always keep a direct model fallback as well
            if self.roboflow_workflow is None:
                # License Plate Detection model from Roboflow Universe (fallback)
                project = rf.workspace().project("lpd-pfzpe")
                self.roboflow_plate_model = project.version(5).model
                logger.info("Roboflow plate detection model initialized (99.5% mAP)")
        except Exception as e:
            logger.warning(f"Failed to initialize Roboflow: {str(e)}")
            logger.info("Using fallback plate detection method")
    
    def detect_vehicles(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect vehicles in an image using YOLOv8.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List[DetectionResult]: List of detected vehicles
        """
        if self.vehicle_model is None:
            logger.error("Vehicle model not loaded")
            return []
        
        results = []
        
        try:
            # Run inference
            predictions = self.vehicle_model(image, verbose=False)[0]
            
            for box in predictions.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter for vehicle classes only
                if class_id in self.VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Calculate aspect ratio for vehicle type correction
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / max(height, 1)
                    
                    # Get initial classification
                    vehicle_class = self.VEHICLE_CLASSES[class_id]
                    
                    # =====================================================
                    # HEURISTIC: Correct truck/car misclassification
                    # =====================================================
                    # Trucks (class 7) are often confused with cars (class 2)
                    # Real trucks typically:
                    # - Have aspect ratio > 1.8 (wider than tall) for front view
                    # - Or are much larger (occupy more of frame)
                    # Sedans typically have aspect ratio 1.0-1.6 for front view
                    
                    if class_id == 7:  # Classified as truck
                        # If aspect ratio suggests it's a car (front view of sedan)
                        # and confidence isn't very high, reclassify as car
                        if aspect_ratio < 1.8 and confidence < 0.75:
                            vehicle_class = "car"
                            class_id = 2
                            logger.debug(f"Reclassified truck -> car (aspect ratio: {aspect_ratio:.2f}, conf: {confidence:.2f})")
                        # Also check if it's a profile view (tall and narrow) - likely a car
                        elif aspect_ratio < 1.3:
                            vehicle_class = "car"
                            class_id = 2
                            logger.debug(f"Reclassified truck -> car (profile view, AR: {aspect_ratio:.2f})")
                    
                    # Crop vehicle region
                    cropped = image[y1:y2, x1:x2].copy()
                    
                    result = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=vehicle_class,
                        cropped_image=cropped
                    )
                    results.append(result)
            
            logger.info(f"Detected {len(results)} vehicles")
            
        except Exception as e:
            logger.error(f"Vehicle detection error: {str(e)}")
        
        return results
    
    def detect_plates(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image (BGR format), typically a cropped vehicle region
            
        Returns:
            List[DetectionResult]: List of detected license plates
        """
        results = []
        
        try:
            # Priority 1: Use Roboflow API for best accuracy
            if self.roboflow_plate_model is not None:
                results = self._detect_plates_roboflow(image)
                if results:
                    return results
            
            # Priority 2: Use local custom plate detection model
            if self.plate_model is not None:
                # Use custom plate detection model
                predictions = self.plate_model(image, verbose=False)[0]
                
                for box in predictions.boxes:
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Crop plate region
                        cropped = image[y1:y2, x1:x2].copy()
                        
                        result = DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=0,
                            class_name="plate",
                            cropped_image=cropped
                        )
                        results.append(result)
            else:
                # Priority 3: Fallback - Estimate plate region from vehicle image
                plate_region = self._estimate_plate_region(image)
                if plate_region is not None:
                    results.append(plate_region)
            
            logger.info(f"Detected {len(results)} license plates")
            
        except Exception as e:
            logger.error(f"Plate detection error: {str(e)}")
        
        return results
    
    def _detect_plates_roboflow(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect license plates using Roboflow API (Workflow preferred if configured).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List[DetectionResult]: List of detected license plates
        """
        results = []
        
        try:
            # Save image temporarily for Roboflow API
            import tempfile
            import time

            # Create temp file, close it, then write to it
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(tmp_fd)  # Close file descriptor immediately

            try:
                cv2.imwrite(tmp_path, image)

                prediction = None
                # Prefer workflow if available
                if self.roboflow_workflow is not None:
                    # Try common SDK call signatures for workflows
                    for method_name, kwargs in (
                        ("predict", {"image": tmp_path}),
                        ("predict", {"image_path": tmp_path}),
                        ("run", {"image": tmp_path}),
                        ("infer", {"image": tmp_path}),
                    ):
                        if hasattr(self.roboflow_workflow, method_name):
                            try:
                                prediction = getattr(self.roboflow_workflow, method_name)(**kwargs)
                                # Some SDKs return objects with .json() or dict
                                if hasattr(prediction, "json"):
                                    prediction = prediction.json()
                                break
                            except Exception:
                                prediction = None

                # Fallback to direct model API
                if prediction is None and self.roboflow_plate_model is not None:
                    prediction = self.roboflow_plate_model.predict(tmp_path, confidence=40).json()
            finally:
                # Clean up temp file with retry for Windows
                for _ in range(3):
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        break
                    except OSError:
                        time.sleep(0.1)

            if prediction is None:
                return results

            # Process predictions (support both direct model and workflow outputs)
            for pred in prediction.get('predictions', []):
                x = int(pred['x'] - pred['width'] / 2)
                y = int(pred['y'] - pred['height'] / 2)
                w = int(pred['width'])
                h = int(pred['height'])
                confidence = pred['confidence']
                # Ensure coordinates are within bounds with padding for OCR
                img_height, img_width = image.shape[:2]

                # Add 10% padding around plate for better OCR
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)

                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                x2 = min(img_width, x + w + 2 * pad_x)
                y2 = min(img_height, y + h + 2 * pad_y)

                # Crop plate region with padding
                cropped = image[y:y2, x:x2].copy()
                
                result = DetectionResult(
                    bbox=(x, y, x2, y2),
                    confidence=confidence,
                    class_id=0,
                    class_name="plate",
                    cropped_image=cropped
                )
                results.append(result)
                logger.info(f"Roboflow detected plate with {confidence:.2%} confidence")

            # Some workflows return nested structures. Try to search recursively for plate-like predictions
            if not results:
                def _collect_preds(obj):
                    found = []
                    if isinstance(obj, dict):
                        if {'x','y','width','height'}.issubset(set(obj.keys())):
                            found.append(obj)
                        for v in obj.values():
                            found.extend(_collect_preds(v))
                    elif isinstance(obj, list):
                        for it in obj:
                            found.extend(_collect_preds(it))
                    return found

                all_preds = _collect_preds(prediction)
                for pred in all_preds:
                    try:
                        x = int(pred['x'] - pred['width'] / 2)
                        y = int(pred['y'] - pred['height'] / 2)
                        w = int(pred['width'])
                        h = int(pred['height'])
                        confidence = float(pred.get('confidence', 0.5))
                        img_height, img_width = image.shape[:2]
                        pad_x = int(w * 0.1)
                        pad_y = int(h * 0.1)
                        x = max(0, x - pad_x)
                        y = max(0, y - pad_y)
                        x2 = min(img_width, x + w + 2 * pad_x)
                        y2 = min(img_height, y + h + 2 * pad_y)
                        cropped = image[y:y2, x:x2].copy()
                        results.append(DetectionResult(
                            bbox=(x, y, x2, y2),
                            confidence=confidence,
                            class_id=0,
                            class_name="plate",
                            cropped_image=cropped
                        ))
                    except Exception:
                        continue
            
        except Exception as e:
            logger.error(f"Roboflow plate detection error: {str(e)}")
        
        return results
    
    def _estimate_plate_region(self, vehicle_image: np.ndarray) -> Optional[DetectionResult]:
        """
        Estimate license plate region using image processing when model unavailable.
        Assumes plate is in lower portion of vehicle image.
        
        Args:
            vehicle_image: Cropped vehicle image
            
        Returns:
            Optional[DetectionResult]: Estimated plate region or None
        """
        try:
            height, width = vehicle_image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
            
            plate_contour = None
            
            for contour in contours:
                # Approximate contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # License plates are typically rectangular (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Nigerian plates aspect ratio is typically between 2:1 and 5:1
                    if 1.5 <= aspect_ratio <= 6.0:
                        # Plate should be in lower half of vehicle
                        if y > height * 0.3:
                            plate_contour = (x, y, w, h)
                            break
            
            if plate_contour:
                x, y, w, h = plate_contour
                cropped = vehicle_image[y:y+h, x:x+w].copy()
                
                return DetectionResult(
                    bbox=(x, y, x+w, y+h),
                    confidence=0.6,  # Lower confidence for estimated region
                    class_id=0,
                    class_name="plate",
                    cropped_image=cropped
                )
            
            # Last resort: crop lower center portion as likely plate region
            plate_y = int(height * 0.6)
            plate_h = int(height * 0.25)
            plate_x = int(width * 0.2)
            plate_w = int(width * 0.6)
            
            cropped = vehicle_image[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w].copy()
            
            return DetectionResult(
                bbox=(plate_x, plate_y, plate_x+plate_w, plate_y+plate_h),
                confidence=0.4,
                class_id=0,
                class_name="plate",
                cropped_image=cropped
            )
            
        except Exception as e:
            logger.error(f"Plate region estimation error: {str(e)}")
            return None
    
    def detect_full_pipeline(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run full detection pipeline: vehicles -> plates.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List[Dict]: List of detection results with vehicle and plate info
        """
        pipeline_results = []
        
        # Stage 1: Detect vehicles
        vehicles = self.detect_vehicles(image)
        
        for vehicle in vehicles:
            result = {
                "vehicle": vehicle.to_dict(),
                "vehicle_type": vehicle.class_name,
                "vehicle_confidence": vehicle.confidence,
                "plates": []
            }
            
            # Stage 2: Detect plates within vehicle region
            if vehicle.cropped_image is not None:
                plates = self.detect_plates(vehicle.cropped_image)
                
                for plate in plates:
                    # Adjust plate coordinates relative to original image
                    vx1, vy1, vx2, vy2 = vehicle.bbox
                    px1, py1, px2, py2 = plate.bbox
                    
                    global_bbox = (
                        vx1 + px1,
                        vy1 + py1,
                        vx1 + px2,
                        vy1 + py2
                    )
                    
                    plate_info = {
                        "bbox": global_bbox,
                        "local_bbox": plate.bbox,
                        "confidence": plate.confidence,
                        "cropped_image": plate.cropped_image
                    }
                    result["plates"].append(plate_info)
            
            pipeline_results.append(result)
        
        return pipeline_results
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        plate_texts: List[str] = None,
        is_stolen: List[bool] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: Detection results from detect_full_pipeline
            plate_texts: List of OCR extracted plate texts
            is_stolen: List of boolean flags for stolen status
            
        Returns:
            np.ndarray: Image with drawn annotations
        """
        output = image.copy()
        plate_texts = plate_texts or []
        is_stolen = is_stolen or []
        
        plate_idx = 0
        
        for detection in detections:
            vehicle = detection["vehicle"]
            x1, y1, x2, y2 = vehicle["bbox"]
            
            # Determine color based on stolen status
            color = (0, 255, 0)  # Green default
            label = f"{vehicle['class_name']} ({vehicle['confidence']:.2f})"
            
            # Check if any plate from this vehicle is stolen
            for plate in detection["plates"]:
                if plate_idx < len(is_stolen) and is_stolen[plate_idx]:
                    color = (0, 0, 255)  # Red for stolen
                    if plate_idx < len(plate_texts):
                        label = f"STOLEN: {plate_texts[plate_idx]}"
                elif plate_idx < len(plate_texts) and plate_texts[plate_idx]:
                    label = f"{vehicle['class_name']}: {plate_texts[plate_idx]}"
                plate_idx += 1
            
            # Draw vehicle bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                output, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width + 10, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                output, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Draw plate bounding boxes
            for plate in detection["plates"]:
                px1, py1, px2, py2 = plate["bbox"]
                cv2.rectangle(output, (px1, py1), (px2, py2), (255, 0, 0), 2)
        
        return output


# Singleton instance
_detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """
    Get or create singleton detection service instance.
    
    Returns:
        DetectionService: The detection service instance
    """
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService(
            confidence_threshold=settings.DETECTION_CONFIDENCE
        )
    return _detection_service
