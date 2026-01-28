"""
Services package initialization.
Exports all service modules.

Enhanced modules for BSc research project:
- tracking: SORT-based object tracking
- roi_processor: Region of interest extraction
- image_quality: Quality assessment and enhancement
- plate_validator: Nigerian plate format validation
- confidence_pipeline: Adaptive processing pipeline
- plate_geometry: Perspective and rotation correction for angled plates
- color_detection: Vehicle color detection using HSV analysis
- email_service: Email alerts for stolen vehicle detections
"""

from services.detection import DetectionService, get_detection_service, DetectionResult
from services.ocr import OCRService, get_ocr_service
# Use workflow processor (Roboflow API) instead of local processing
from services.workflow_processor import WorkflowVideoProcessor, get_workflow_processor, get_video_processor, FrameResult
from services.roboflow_workflow import RoboflowWorkflowService, get_workflow_service
from services.database_service import StolenVehicleService, DetectionLogService

# Enhanced modules
from services.tracking import SORTTracker, get_tracker, reset_tracker, TrackedObject
from services.roi_processor import ROIProcessor, get_roi_processor, ROIConfig
from services.image_quality import ImageQualityProcessor, get_quality_processor, QualityMetrics, QualityConfig
from services.plate_validator import NigerianPlateValidator, process_ocr_result, get_validator
from services.confidence_pipeline import ConfidenceDrivenPipeline, get_pipeline, reset_pipeline, PipelineConfig
from services.plate_geometry import PlateGeometryCorrector, get_geometry_corrector, PlateGeometry, CorrectionConfig

# New feature modules
from services.color_detection import ColorDetectionService, get_color_service
from services.email_service import EmailService, get_email_service

__all__ = [
    # Core services
    "DetectionService", "get_detection_service", "DetectionResult",
    "OCRService", "get_ocr_service",
    "WorkflowVideoProcessor", "get_workflow_processor", "get_video_processor", "FrameResult",
    "RoboflowWorkflowService", "get_workflow_service",
    "StolenVehicleService", "DetectionLogService",
    # Enhanced modules
    "SORTTracker", "get_tracker", "reset_tracker", "TrackedObject",
    "ROIProcessor", "get_roi_processor", "ROIConfig",
    "ImageQualityProcessor", "get_quality_processor", "QualityMetrics", "QualityConfig",
    "NigerianPlateValidator", "process_ocr_result", "get_validator",
    "ConfidenceDrivenPipeline", "get_pipeline", "reset_pipeline", "PipelineConfig",
    "PlateGeometryCorrector", "get_geometry_corrector", "PlateGeometry", "CorrectionConfig",
    # New feature modules
    "ColorDetectionService", "get_color_service",
    "EmailService", "get_email_service",
]

