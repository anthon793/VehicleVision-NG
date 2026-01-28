/**
 * Detection Overlay Component
 * Draws bounding boxes and labels on video feed
 */

import React, { useEffect, useRef } from 'react';

const DetectionOverlay = ({ detections, width, height, containerRef }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !detections || !width || !height) return;

    const container = containerRef?.current;
    if (!container) return;

    // Set canvas size to match container
    const containerRect = container.getBoundingClientRect();
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale factors
    const scaleX = canvas.width / width;
    const scaleY = canvas.height / height;

    detections.forEach((detection) => {
      const { bbox, plate_text, is_stolen, vehicle_type, confidence } = detection;
      
      // Scale coordinates
      const x1 = bbox.x1 * scaleX;
      const y1 = bbox.y1 * scaleY;
      const x2 = bbox.x2 * scaleX;
      const y2 = bbox.y2 * scaleY;
      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      // Set colors based on stolen status
      const color = is_stolen ? '#ef4444' : '#22c55e';
      const bgColor = is_stolen ? 'rgba(239, 68, 68, 0.9)' : 'rgba(34, 197, 94, 0.9)';

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, boxWidth, boxHeight);

      // Draw corner accents
      const cornerLength = Math.min(20, boxWidth / 4, boxHeight / 4);
      ctx.lineWidth = 4;
      
      // Top-left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerLength);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerLength, y1);
      ctx.stroke();
      
      // Top-right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLength, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cornerLength);
      ctx.stroke();
      
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - cornerLength);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + cornerLength, y2);
      ctx.stroke();
      
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLength, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - cornerLength);
      ctx.stroke();

      // Draw label
      const label = is_stolen 
        ? `⚠️ STOLEN: ${plate_text}` 
        : `${vehicle_type || 'Vehicle'}: ${plate_text || 'Unknown'}`;
      
      ctx.font = 'bold 14px system-ui, sans-serif';
      const textMetrics = ctx.measureText(label);
      const textHeight = 20;
      const padding = 6;
      const labelWidth = textMetrics.width + padding * 2;
      const labelHeight = textHeight + padding;
      
      // Label background
      ctx.fillStyle = bgColor;
      ctx.fillRect(x1, y1 - labelHeight - 4, labelWidth, labelHeight);
      
      // Label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + padding, y1 - padding - 4);

      // Confidence badge
      if (confidence) {
        const confText = `${Math.round(confidence * 100)}%`;
        const confWidth = ctx.measureText(confText).width + 8;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(x2 - confWidth, y2 + 4, confWidth, 18);
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px system-ui, sans-serif';
        ctx.fillText(confText, x2 - confWidth + 4, y2 + 16);
      }
    });
  }, [detections, width, height, containerRef]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 10 }}
    />
  );
};

export default DetectionOverlay;
