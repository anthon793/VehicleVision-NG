/**
 * Custom hook for real-time detection
 * Manages detection loop and results
 * 
 * Enhanced with:
 * - Geolocation support
 * - Enhanced detection endpoint option
 * - Quality metrics display
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { detectionAPI } from '../services/api';

export const useDetection = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [stolenAlert, setStolenAlert] = useState(null);
  const [geolocation, setGeolocation] = useState(null);
  const [geoError, setGeoError] = useState(null);
  const [useEnhancedDetection, setUseEnhancedDetection] = useState(true);
  
  const intervalRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());
  const watchIdRef = useRef(null);

  /**
   * Start watching geolocation
   */
  const startGeolocation = useCallback(() => {
    if (!navigator.geolocation) {
      setGeoError('Geolocation not supported');
      return;
    }

    // Get initial position
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setGeolocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: position.timestamp
        });
        setGeoError(null);
      },
      (error) => {
        console.warn('Geolocation error:', error.message);
        setGeoError(error.message);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );

    // Watch for updates
    watchIdRef.current = navigator.geolocation.watchPosition(
      (position) => {
        setGeolocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: position.timestamp
        });
        setGeoError(null);
      },
      (error) => {
        console.warn('Geolocation watch error:', error.message);
        // Don't clear existing geolocation on error
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 }
    );
  }, []);

  /**
   * Stop watching geolocation
   */
  const stopGeolocation = useCallback(() => {
    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
  }, []);

  /**
   * Process a single frame
   */
  const detectFrame = useCallback(async (imageBase64) => {
    try {
      setError(null);
      const startTime = Date.now();
      
      let response;
      
      if (useEnhancedDetection) {
        // Use enhanced detection endpoint with geolocation
        response = await detectionAPI.detectFrameEnhanced(
          imageBase64,
          true,
          geolocation?.latitude,
          geolocation?.longitude,
          geolocation?.accuracy
        );
      } else {
        // Use standard detection
        response = await detectionAPI.detectFramePost(imageBase64, true);
      }
      
      const endTime = Date.now();
      setLatency(endTime - startTime);
      
      // Update FPS counter
      frameCountRef.current++;
      const now = Date.now();
      const elapsed = now - lastFpsTimeRef.current;
      if (elapsed >= 1000) {
        setFps(Math.round((frameCountRef.current * 1000) / elapsed));
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }
      
      setResults(response);
      
      // Check for stolen vehicles
      if (response.has_stolen) {
        const stolenPlates = response.detections
          .filter(d => d.is_stolen)
          .map(d => d.plate_text);
        setStolenAlert({
          plates: stolenPlates,
          timestamp: new Date().toISOString(),
          geolocation: geolocation
        });
      } else {
        setStolenAlert(null);
      }
      
      return response;
    } catch (err) {
      console.error('Detection error:', err);
      setError(err.response?.data?.detail || 'Detection failed');
      return null;
    }
  }, [geolocation, useEnhancedDetection]);

  /**
   * Start continuous detection loop
   */
  const startDetection = useCallback((captureFunction, intervalMs = 500) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    setIsDetecting(true);
    frameCountRef.current = 0;
    lastFpsTimeRef.current = Date.now();
    
    // Start geolocation tracking
    startGeolocation();
    
    // Initial detection
    const frame = captureFunction();
    if (frame) {
      detectFrame(frame);
    }
    
    // Set up interval
    intervalRef.current = setInterval(async () => {
      const frame = captureFunction();
      if (frame) {
        await detectFrame(frame);
      }
    }, intervalMs);
  }, [detectFrame, startGeolocation]);

  /**
   * Stop continuous detection
   */
  const stopDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsDetecting(false);
    setFps(0);
    setLatency(0);
    stopGeolocation();
  }, [stopGeolocation]);

  /**
   * Clear stolen alert
   */
  const clearAlert = useCallback(() => {
    setStolenAlert(null);
  }, []);

  /**
   * Toggle enhanced detection mode
   */
  const toggleEnhancedDetection = useCallback(() => {
    setUseEnhancedDetection(prev => !prev);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      stopGeolocation();
    };
  }, [stopGeolocation]);

  return {
    isDetecting,
    results,
    error,
    fps,
    latency,
    stolenAlert,
    geolocation,
    geoError,
    useEnhancedDetection,
    detectFrame,
    startDetection,
    stopDetection,
    clearAlert,
    setError,
    toggleEnhancedDetection,
    startGeolocation,
    stopGeolocation
  };
};

export default useDetection;
