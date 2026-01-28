"""
Quick Evaluation Script - Test the model on a few images.

Usage:
    python evaluation/quick_eval.py --images ./test_images

Or test with a single image:
    python evaluation/quick_eval.py --image ./test.jpg --expected-plate ABC123XY
"""

import os
import sys
import cv2
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services import get_video_processor


def test_single_image(image_path: str, expected_plate: str = None):
    """Test detection on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Get processor
    processor = get_video_processor()
    
    # Run detection
    start_time = time.time()
    result = processor.process_image(image, set())
    processing_time = (time.time() - start_time) * 1000
    
    print(f"\nProcessing time: {processing_time:.0f}ms")
    
    # Results
    print(f"\n--- DETECTION RESULTS ---")
    
    if result.get("detections"):
        for i, det in enumerate(result["detections"]):
            print(f"\nVehicle {i+1}:")
            print(f"  Type: {det.get('vehicle_type', 'unknown')}")
            print(f"  Confidence: {det.get('vehicle_confidence', 0):.2%}")
            print(f"  Color: {det.get('vehicle_color', 'unknown')}")
    else:
        print("No vehicles detected")
    
    if result.get("plate_texts"):
        for i, plate in enumerate(result["plate_texts"]):
            print(f"\nPlate {i+1}: {plate}")
            
            if expected_plate:
                expected_normalized = ''.join(c.upper() for c in expected_plate if c.isalnum())
                detected_normalized = ''.join(c.upper() for c in plate if c.isalnum())
                
                if expected_normalized == detected_normalized:
                    print(f"  ✅ CORRECT - Matches expected: {expected_plate}")
                else:
                    print(f"  ❌ INCORRECT")
                    print(f"     Expected: {expected_plate}")
                    print(f"     Got:      {plate}")
    else:
        print("\nNo plates detected")
        if expected_plate:
            print(f"  ❌ MISSED - Expected: {expected_plate}")
    
    return result


def test_directory(images_dir: str):
    """Test all images in a directory."""
    images_dir = Path(images_dir)
    
    # Collect results
    results = {
        "total": 0,
        "vehicles_detected": 0,
        "plates_detected": 0,
        "total_time_ms": 0,
        "images": []
    }
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(images_dir.glob(ext))
    
    if not images:
        print(f"No images found in: {images_dir}")
        return
    
    print(f"\nFound {len(images)} images to test\n")
    
    processor = get_video_processor()
    
    for img_path in images:
        results["total"] += 1
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"SKIP: Could not load: {img_path.name}")
            continue
        
        start_time = time.time()
        result = processor.process_image(image, set())
        processing_time = (time.time() - start_time) * 1000
        results["total_time_ms"] += processing_time
        
        # Check results
        vehicle_detected = bool(result.get("detections"))
        plate_detected = bool(result.get("plate_texts"))
        plate_text = result["plate_texts"][0] if plate_detected else "N/A"
        vehicle_type = result["detections"][0].get("vehicle_type") if vehicle_detected else "N/A"
        
        if vehicle_detected:
            results["vehicles_detected"] += 1
        if plate_detected:
            results["plates_detected"] += 1
        
        status = "✅" if plate_detected else "❌"
        print(f"{status} {img_path.name}: {vehicle_type} | Plate: {plate_text} | {processing_time:.0f}ms")
        
        results["images"].append({
            "file": img_path.name,
            "vehicle_detected": vehicle_detected,
            "plate_detected": plate_detected,
            "plate_text": plate_text,
            "vehicle_type": vehicle_type,
            "time_ms": processing_time
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total Images:        {results['total']}")
    print(f"Vehicles Detected:   {results['vehicles_detected']} ({results['vehicles_detected']/results['total']*100:.1f}%)")
    print(f"Plates Detected:     {results['plates_detected']} ({results['plates_detected']/results['total']*100:.1f}%)")
    print(f"Avg Processing Time: {results['total_time_ms']/results['total']:.0f}ms")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick model evaluation")
    parser.add_argument("--image", type=str, help="Single image to test")
    parser.add_argument("--images", type=str, help="Directory of images to test")
    parser.add_argument("--expected-plate", type=str, help="Expected plate number (for single image)")
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image, args.expected_plate)
    elif args.images:
        test_directory(args.images)
    else:
        print("Usage:")
        print("  Test single image:  python quick_eval.py --image ./test.jpg --expected-plate ABC123")
        print("  Test directory:     python quick_eval.py --images ./test_images/")


if __name__ == "__main__":
    main()
