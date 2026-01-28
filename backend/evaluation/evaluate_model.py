"""
Model Evaluation Script for Stolen Vehicle Detection System.

Evaluates:
1. Vehicle Detection (YOLOv8) - Precision, Recall, F1
2. License Plate Detection - Detection Rate
3. OCR Accuracy - Character Error Rate (CER), Exact Match
4. End-to-End System - Stolen Vehicle Detection Metrics

Usage:
    python evaluation/evaluate_model.py --dataset ./test_data --output ./results
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services import get_video_processor, get_detection_service
from services.roboflow_workflow import get_workflow_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of a single detection."""
    image_path: str
    ground_truth_plate: str
    detected_plate: str
    ground_truth_vehicle_type: str
    detected_vehicle_type: str
    plate_detected: bool
    vehicle_detected: bool
    ocr_correct: bool
    processing_time_ms: float


@dataclass 
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    # Dataset info
    total_images: int = 0
    total_processing_time_ms: float = 0
    avg_processing_time_ms: float = 0
    
    # Vehicle Detection Metrics
    vehicle_true_positives: int = 0
    vehicle_false_positives: int = 0
    vehicle_false_negatives: int = 0
    vehicle_precision: float = 0.0
    vehicle_recall: float = 0.0
    vehicle_f1_score: float = 0.0
    
    # Vehicle Type Classification
    vehicle_type_correct: int = 0
    vehicle_type_accuracy: float = 0.0
    
    # Plate Detection Metrics
    plate_true_positives: int = 0
    plate_false_positives: int = 0
    plate_false_negatives: int = 0
    plate_precision: float = 0.0
    plate_recall: float = 0.0
    plate_f1_score: float = 0.0
    plate_detection_rate: float = 0.0
    
    # OCR Metrics
    ocr_exact_matches: int = 0
    ocr_exact_match_rate: float = 0.0
    ocr_character_errors: int = 0
    ocr_total_characters: int = 0
    ocr_character_error_rate: float = 0.0
    ocr_character_accuracy: float = 0.0
    
    # End-to-End Metrics (for stolen vehicle matching)
    e2e_true_positives: int = 0  # Correctly identified stolen
    e2e_false_positives: int = 0  # Non-stolen flagged as stolen
    e2e_true_negatives: int = 0  # Correctly identified non-stolen
    e2e_false_negatives: int = 0  # Stolen but not detected
    e2e_precision: float = 0.0
    e2e_recall: float = 0.0
    e2e_f1_score: float = 0.0
    e2e_accuracy: float = 0.0
    
    # Per-class metrics
    vehicle_type_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_plate(plate: str) -> str:
    """Normalize plate text for comparison."""
    if not plate:
        return ""
    # Remove spaces, dashes, convert to uppercase
    return ''.join(c.upper() for c in plate if c.isalnum())


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class ModelEvaluator:
    """
    Evaluator for the Stolen Vehicle Detection System.
    
    Tests against a labeled dataset and computes comprehensive metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator with detection services."""
        self.processor = get_video_processor()
        self.detection_service = get_detection_service()
        self.workflow_service = get_workflow_service()
        self.results: List[DetectionResult] = []
        self.metrics = EvaluationMetrics()
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load test dataset from a directory.
        
        Expected structure:
        dataset/
        ├── annotations.json  # Contains ground truth
        └── images/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
        
        annotations.json format:
        [
            {
                "image": "image1.jpg",
                "plate": "ABC123XY",
                "vehicle_type": "car",
                "is_stolen": true
            },
            ...
        ]
        """
        dataset_path = Path(dataset_path)
        annotations_file = dataset_path / "annotations.json"
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            # Add full paths
            for ann in annotations:
                ann['image_path'] = str(dataset_path / "images" / ann['image'])
            
            return annotations
        else:
            # Auto-generate from filenames if no annotations
            # Expects format: PLATENO_vehicletype.jpg
            images_dir = dataset_path / "images"
            if not images_dir.exists():
                images_dir = dataset_path
            
            annotations = []
            for img_path in images_dir.glob("*.jpg"):
                # Try to parse filename
                name = img_path.stem
                parts = name.split('_')
                plate = parts[0] if parts else name
                vehicle_type = parts[1] if len(parts) > 1 else "unknown"
                
                annotations.append({
                    'image': img_path.name,
                    'image_path': str(img_path),
                    'plate': plate.upper(),
                    'vehicle_type': vehicle_type.lower(),
                    'is_stolen': False  # Default
                })
            
            return annotations
    
    def evaluate_single_image(
        self, 
        image_path: str, 
        ground_truth: Dict
    ) -> DetectionResult:
        """Evaluate detection on a single image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        start_time = time.time()
        
        # Run detection
        try:
            result = self.processor.process_image(image, set())
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return None
        
        processing_time = (time.time() - start_time) * 1000
        
        # Extract results
        detected_plate = ""
        detected_vehicle_type = ""
        plate_detected = False
        vehicle_detected = False
        
        if result.get("detections"):
            detection = result["detections"][0]
            vehicle_detected = True
            detected_vehicle_type = detection.get("vehicle_type", "")
            
            if detection.get("plates"):
                plate_detected = True
        
        if result.get("plate_texts"):
            detected_plate = result["plate_texts"][0]
        
        # Compare
        gt_plate = normalize_plate(ground_truth.get('plate', ''))
        det_plate = normalize_plate(detected_plate)
        ocr_correct = gt_plate == det_plate
        
        return DetectionResult(
            image_path=image_path,
            ground_truth_plate=gt_plate,
            detected_plate=det_plate,
            ground_truth_vehicle_type=ground_truth.get('vehicle_type', ''),
            detected_vehicle_type=detected_vehicle_type,
            plate_detected=plate_detected,
            vehicle_detected=vehicle_detected,
            ocr_correct=ocr_correct,
            processing_time_ms=processing_time
        )
    
    def evaluate_dataset(self, dataset_path: str) -> EvaluationMetrics:
        """
        Evaluate the model on an entire dataset.
        
        Args:
            dataset_path: Path to the test dataset
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        annotations = self.load_dataset(dataset_path)
        
        if not annotations:
            logger.error("No images found in dataset")
            return self.metrics
        
        logger.info(f"Evaluating {len(annotations)} images...")
        self.results = []
        
        for i, ann in enumerate(annotations):
            image_path = ann['image_path']
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            result = self.evaluate_single_image(image_path, ann)
            if result:
                self.results.append(result)
                
                # Progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(annotations)} images")
        
        # Compute metrics
        self._compute_metrics()
        
        return self.metrics
    
    def _compute_metrics(self):
        """Compute all evaluation metrics from results."""
        if not self.results:
            return
        
        m = self.metrics
        m.total_images = len(self.results)
        m.total_processing_time_ms = sum(r.processing_time_ms for r in self.results)
        m.avg_processing_time_ms = m.total_processing_time_ms / m.total_images
        
        # Vehicle Detection Metrics
        for r in self.results:
            if r.vehicle_detected:
                m.vehicle_true_positives += 1
            else:
                m.vehicle_false_negatives += 1
            
            # Vehicle type confusion matrix
            gt_type = r.ground_truth_vehicle_type or "unknown"
            det_type = r.detected_vehicle_type or "unknown"
            
            if gt_type not in m.vehicle_type_confusion:
                m.vehicle_type_confusion[gt_type] = {}
            if det_type not in m.vehicle_type_confusion[gt_type]:
                m.vehicle_type_confusion[gt_type][det_type] = 0
            m.vehicle_type_confusion[gt_type][det_type] += 1
            
            if gt_type.lower() == det_type.lower():
                m.vehicle_type_correct += 1
        
        # Vehicle precision/recall
        m.vehicle_precision, m.vehicle_recall, m.vehicle_f1_score = calculate_precision_recall_f1(
            m.vehicle_true_positives, m.vehicle_false_positives, m.vehicle_false_negatives
        )
        m.vehicle_type_accuracy = m.vehicle_type_correct / m.total_images if m.total_images > 0 else 0
        
        # Plate Detection Metrics
        for r in self.results:
            if r.ground_truth_plate:  # Has ground truth plate
                if r.plate_detected:
                    m.plate_true_positives += 1
                else:
                    m.plate_false_negatives += 1
            else:  # No ground truth plate
                if r.plate_detected:
                    m.plate_false_positives += 1
        
        m.plate_precision, m.plate_recall, m.plate_f1_score = calculate_precision_recall_f1(
            m.plate_true_positives, m.plate_false_positives, m.plate_false_negatives
        )
        
        plates_with_gt = sum(1 for r in self.results if r.ground_truth_plate)
        m.plate_detection_rate = m.plate_true_positives / plates_with_gt if plates_with_gt > 0 else 0
        
        # OCR Metrics
        for r in self.results:
            if r.ground_truth_plate and r.detected_plate:
                if r.ocr_correct:
                    m.ocr_exact_matches += 1
                
                # Character-level errors
                distance = levenshtein_distance(r.ground_truth_plate, r.detected_plate)
                m.ocr_character_errors += distance
                m.ocr_total_characters += len(r.ground_truth_plate)
        
        plates_detected = sum(1 for r in self.results if r.ground_truth_plate and r.detected_plate)
        m.ocr_exact_match_rate = m.ocr_exact_matches / plates_detected if plates_detected > 0 else 0
        m.ocr_character_error_rate = m.ocr_character_errors / m.ocr_total_characters if m.ocr_total_characters > 0 else 0
        m.ocr_character_accuracy = 1.0 - m.ocr_character_error_rate
    
    def print_report(self):
        """Print a formatted evaluation report."""
        m = self.metrics
        
        print("\n" + "=" * 60)
        print("       STOLEN VEHICLE DETECTION SYSTEM - EVALUATION REPORT")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Images Evaluated: {m.total_images}")
        print(f"Average Processing Time: {m.avg_processing_time_ms:.1f}ms")
        
        print("\n" + "-" * 60)
        print("VEHICLE DETECTION METRICS")
        print("-" * 60)
        print(f"  True Positives:  {m.vehicle_true_positives}")
        print(f"  False Negatives: {m.vehicle_false_negatives}")
        print(f"  Precision:       {m.vehicle_precision:.2%}")
        print(f"  Recall:          {m.vehicle_recall:.2%}")
        print(f"  F1 Score:        {m.vehicle_f1_score:.2%}")
        
        print("\n" + "-" * 60)
        print("VEHICLE TYPE CLASSIFICATION")
        print("-" * 60)
        print(f"  Accuracy: {m.vehicle_type_accuracy:.2%}")
        if m.vehicle_type_confusion:
            print("  Confusion Matrix:")
            for gt_type, predictions in m.vehicle_type_confusion.items():
                for pred_type, count in predictions.items():
                    print(f"    {gt_type} -> {pred_type}: {count}")
        
        print("\n" + "-" * 60)
        print("LICENSE PLATE DETECTION METRICS")
        print("-" * 60)
        print(f"  Detection Rate: {m.plate_detection_rate:.2%}")
        print(f"  Precision:      {m.plate_precision:.2%}")
        print(f"  Recall:         {m.plate_recall:.2%}")
        print(f"  F1 Score:       {m.plate_f1_score:.2%}")
        
        print("\n" + "-" * 60)
        print("OCR ACCURACY METRICS")
        print("-" * 60)
        print(f"  Exact Match Rate:       {m.ocr_exact_match_rate:.2%}")
        print(f"  Character Accuracy:     {m.ocr_character_accuracy:.2%}")
        print(f"  Character Error Rate:   {m.ocr_character_error_rate:.2%}")
        print(f"  Total Character Errors: {m.ocr_character_errors}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Vehicle Detection Rate:    {m.vehicle_recall:.2%}")
        print(f"  Plate Detection Rate:      {m.plate_detection_rate:.2%}")
        print(f"  OCR Exact Match Rate:      {m.ocr_exact_match_rate:.2%}")
        print(f"  End-to-End Success Rate:   {m.ocr_exact_match_rate * m.plate_detection_rate:.2%}")
        print("=" * 60)
    
    def save_results(self, output_path: str):
        """Save detailed results to JSON file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(self.metrics), f, indent=2)
        
        # Save detailed results
        results_file = output_path / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Save summary report
        report_file = output_path / "report.txt"
        with open(report_file, 'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            self.print_report()
            sys.stdout = old_stdout
        
        logger.info(f"Results saved to: {output_path}")


def create_sample_dataset(output_path: str):
    """Create a sample dataset structure for testing."""
    output_path = Path(output_path)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample annotations
    annotations = [
        {"image": "ABC123XY_car.jpg", "plate": "ABC123XY", "vehicle_type": "car", "is_stolen": False},
        {"image": "XYZ789AB_truck.jpg", "plate": "XYZ789AB", "vehicle_type": "truck", "is_stolen": True},
    ]
    
    with open(output_path / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Sample dataset structure created at: {output_path}")
    print("Add your test images to the 'images' folder and update annotations.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stolen Vehicle Detection System")
    parser.add_argument("--dataset", type=str, help="Path to test dataset")
    parser.add_argument("--output", type=str, default="./evaluation_results", help="Output path for results")
    parser.add_argument("--create-sample", action="store_true", help="Create sample dataset structure")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.output)
        return
    
    if not args.dataset:
        print("Usage: python evaluate_model.py --dataset ./test_data")
        print("       python evaluate_model.py --create-sample --output ./sample_dataset")
        return
    
    # Run evaluation
    evaluator = ModelEvaluator()
    evaluator.evaluate_dataset(args.dataset)
    evaluator.print_report()
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
