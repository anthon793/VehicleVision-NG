/**
 * Custom hook for camera access using WebRTC
 * Manages camera stream and frame capture
 */

import { useState, useRef, useCallback, useEffect } from 'react';

export const useCamera = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [currentDevice, setCurrentDevice] = useState(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  // Get available camera devices
  const getDevices = useCallback(async () => {
    try {
      const mediaDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = mediaDevices.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      return videoDevices;
    } catch (err) {
      setError('Failed to enumerate devices');
      return [];
    }
  }, []);

  // Start camera stream
  const startCamera = useCallback(async (deviceId = null) => {
    try {
      setError(null);
      
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment' // Prefer back camera on mobile
        },
        audio: false
      };

      if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Get the actual device being used
      const videoTrack = stream.getVideoTracks()[0];
      setCurrentDevice(videoTrack.getSettings().deviceId);
      
      setIsStreaming(true);
      return true;
    } catch (err) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError') {
        setError('Camera access denied. Please allow camera access in browser settings.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found on this device.');
      } else {
        setError(`Camera error: ${err.message}`);
      }
      setIsStreaming(false);
      return false;
    }
  }, []);

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  // Capture current frame as base64
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !isStreaming) {
      return null;
    }

    const video = videoRef.current;
    
    // Create or reuse canvas
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
    
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Return base64 encoded image (JPEG for smaller size)
    return canvas.toDataURL('image/jpeg', 0.8);
  }, [isStreaming]);

  // Switch to different camera
  const switchCamera = useCallback(async (deviceId) => {
    if (isStreaming) {
      stopCamera();
    }
    return await startCamera(deviceId);
  }, [isStreaming, startCamera, stopCamera]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    isStreaming,
    error,
    devices,
    currentDevice,
    getDevices,
    startCamera,
    stopCamera,
    captureFrame,
    switchCamera,
    setError
  };
};

export default useCamera;
