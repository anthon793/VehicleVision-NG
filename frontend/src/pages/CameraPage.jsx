/**
 * Camera Detection Page Component
 * Live camera feed with real-time vehicle/plate detection
 * 
 * Enhanced with:
 * - Geolocation capture
 * - Quality metrics display
 * - Enhanced detection mode toggle
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import DetectionOverlay from '../components/DetectionOverlay';
import StolenAlert from '../components/StolenAlert';
import LoadingSpinner from '../components/LoadingSpinner';
import { useCamera } from '../hooks/useCamera';
import { useDetection } from '../hooks/useDetection';
import {
  Camera, CameraOff, Play, Square, RefreshCw,
  Settings, Zap, Clock, AlertTriangle, Monitor,
  MapPin, Gauge, Shield, RotateCw
} from 'lucide-react';

const CameraPage = () => {
  const containerRef = useRef(null);
  const [showSettings, setShowSettings] = useState(false);
  const [detectionInterval, setDetectionInterval] = useState(500);

  const {
    videoRef,
    isStreaming,
    error: cameraError,
    devices,
    currentDevice,
    getDevices,
    startCamera,
    stopCamera,
    captureFrame,
    switchCamera
  } = useCamera();

  const {
    isDetecting,
    results,
    error: detectionError,
    fps,
    latency,
    stolenAlert,
    geolocation,
    geoError,
    useEnhancedDetection,
    startDetection,
    stopDetection,
    clearAlert,
    toggleEnhancedDetection
  } = useDetection();

  // Get available cameras on mount
  useEffect(() => {
    getDevices();
  }, [getDevices]);

  // Handle start/stop camera
  const handleToggleCamera = async () => {
    if (isStreaming) {
      stopDetection();
      stopCamera();
    } else {
      await startCamera();
    }
  };

  // Handle start/stop detection
  const handleToggleDetection = () => {
    if (isDetecting) {
      stopDetection();
    } else {
      startDetection(captureFrame, detectionInterval);
    }
  };

  // Handle camera switch
  const handleSwitchCamera = async (deviceId) => {
    if (isDetecting) {
      stopDetection();
    }
    await switchCamera(deviceId);
  };

  return (
    <>
      <div className="p-4 lg:p-6">
        <div className="mb-4">
          <h1 className="text-xl font-bold text-gray-900">Live Camera Detection</h1>
          <p className="text-sm text-gray-600">Real-time vehicle and license plate detection from your camera</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Camera Feed */}
          <div className="lg:col-span-2">
            <div className="card p-0 overflow-hidden">
              {/* Camera Container */}
              <div ref={containerRef} className="camera-container bg-gray-900">
                <video
                  ref={videoRef}
                  className="w-full h-full object-contain"
                  autoPlay
                  playsInline
                  muted
                />

                {/* Detection Overlay */}
                {isDetecting && results?.detections && (
                  <DetectionOverlay
                    detections={results.detections}
                    width={results.frame_width}
                    height={results.frame_height}
                    containerRef={containerRef}
                  />
                )}

                {/* No Camera Message */}
                {!isStreaming && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                    <div className="text-center text-gray-400">
                      <CameraOff className="h-12 w-12 mx-auto mb-3" />
                      <p className="text-sm font-medium">Camera not active</p>
                      <p className="text-xs">Click "Start Camera" to begin</p>
                    </div>
                  </div>
                )}

                {/* Stats Overlay */}
                {isDetecting && (
                  <div className="absolute top-3 left-3 bg-black/70 text-white px-2.5 py-1.5 rounded text-xs space-y-0.5">
                    <div className="flex items-center space-x-1.5">
                      <Zap className="h-3.5 w-3.5 text-yellow-400" />
                      <span>{fps} FPS</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <Clock className="h-3.5 w-3.5 text-blue-400" />
                      <span>{latency}ms latency</span>
                    </div>
                  </div>
                )}

                {/* Detection Status */}
                {isDetecting && (
                  <div className="absolute top-3 right-3">
                    <div className="flex items-center space-x-1.5 bg-success-500 text-white px-2 py-0.5 rounded-full text-xs">
                      <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                      <span>Detecting</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="p-3 bg-gray-50 border-t border-gray-200">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={handleToggleCamera}
                      className={`btn ${isStreaming ? 'btn-danger' : 'btn-primary'} flex items-center space-x-1.5`}
                    >
                      {isStreaming ? (
                        <>
                          <CameraOff className="h-4 w-4" />
                          <span>Stop Camera</span>
                        </>
                      ) : (
                        <>
                          <Camera className="h-4 w-4" />
                          <span>Start Camera</span>
                        </>
                      )}
                    </button>

                    {isStreaming && (
                      <button
                        onClick={handleToggleDetection}
                        className={`btn ${isDetecting ? 'btn-secondary' : 'btn-success'} flex items-center space-x-1.5`}
                      >
                        {isDetecting ? (
                          <>
                            <Square className="h-4 w-4" />
                            <span>Stop Detection</span>
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4" />
                            <span>Start Detection</span>
                          </>
                        )}
                      </button>
                    )}
                  </div>

                  <button
                    onClick={() => setShowSettings(!showSettings)}
                    className="btn btn-secondary flex items-center space-x-1.5"
                  >
                    <Settings className="h-4 w-4" />
                    <span>Settings</span>
                  </button>
                </div>

                {/* Settings Panel */}
                {showSettings && (
                  <div className="mt-3 p-3 bg-white rounded border border-gray-200">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">Detection Settings</h3>

                    {/* Camera Selection */}
                    {devices.length > 1 && (
                      <div className="mb-3">
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Camera Device
                        </label>
                        <select
                          value={currentDevice || ''}
                          onChange={(e) => handleSwitchCamera(e.target.value)}
                          className="input"
                        >
                          {devices.map((device) => (
                            <option key={device.deviceId} value={device.deviceId}>
                              {device.label || `Camera ${devices.indexOf(device) + 1}`}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    {/* Detection Interval */}
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Detection Interval: {detectionInterval}ms
                      </label>
                      <input
                        type="range"
                        min="200"
                        max="2000"
                        step="100"
                        value={detectionInterval}
                        onChange={(e) => setDetectionInterval(parseInt(e.target.value))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-0.5">
                        <span>Faster (200ms)</span>
                        <span>Slower (2000ms)</span>
                      </div>
                    </div>

                    {/* Enhanced Detection Toggle */}
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="block text-xs font-medium text-gray-700">
                            Enhanced Detection
                          </label>
                          <p className="text-xs text-gray-500">
                            Tracking, validation & geolocation
                          </p>
                        </div>
                        <button
                          onClick={toggleEnhancedDetection}
                          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${useEnhancedDetection ? 'bg-primary-500' : 'bg-gray-300'
                            }`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${useEnhancedDetection ? 'translate-x-4' : 'translate-x-1'
                              }`}
                          />
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error Messages */}
                {(cameraError || detectionError) && (
                  <div className="mt-3 p-2.5 bg-danger-50 border border-danger-200 rounded text-danger-700 text-xs">
                    {cameraError || detectionError}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Detection Results Panel */}
          <div className="space-y-4">
            {/* Current Detections */}
            <div className="card">
              <h2 className="card-header flex items-center space-x-1.5">
                <Monitor className="h-4 w-4 text-primary-500" />
                <span>Current Detections</span>
              </h2>

              {results?.detections?.length > 0 ? (
                <div className="space-y-2">
                  {results.detections.map((detection, index) => (
                    <div
                      key={index}
                      className={`p-2.5 rounded border ${detection.is_stolen
                          ? 'bg-danger-50 border-danger-200'
                          : 'bg-gray-50 border-gray-200'
                        }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className={`font-mono text-sm font-semibold ${detection.is_stolen ? 'text-danger-700' : 'text-gray-900'
                          }`}>
                          {detection.plate_text || 'Detecting...'}
                        </span>
                        {detection.is_stolen && (
                          <span className="badge badge-danger">STOLEN</span>
                        )}
                      </div>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{detection.vehicle_type || 'Vehicle'}</span>
                        <span>{Math.round(detection.confidence * 100)}% confidence</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-400">
                  <Camera className="h-8 w-8 mx-auto mb-2" />
                  <p className="text-sm">No vehicles detected</p>
                  {!isDetecting && <p className="text-xs">Start detection to scan for vehicles</p>}
                </div>
              )}
            </div>

            {/* Session Statistics */}
            <div className="card">
              <h2 className="card-header">Session Stats</h2>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Status</span>
                  <span className={`badge ${isDetecting ? 'badge-success' : 'badge-warning'}`}>
                    {isDetecting ? 'Active' : 'Idle'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Mode</span>
                  <span className={`badge ${useEnhancedDetection ? 'badge-primary' : 'badge-secondary'}`}>
                    {useEnhancedDetection ? 'Enhanced' : 'Standard'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Frame Rate</span>
                  <span className="font-mono">{fps} FPS</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Latency</span>
                  <span className="font-mono">{latency} ms</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Processing Time</span>
                  <span className="font-mono">{results?.processing_time_ms?.toFixed(0) || 0} ms</span>
                </div>
              </div>
            </div>

            {/* Quality Metrics (when enhanced mode is on) */}
            {useEnhancedDetection && results?.quality_metrics && (
              <div className="card">
                <h2 className="card-header flex items-center space-x-1.5">
                  <Gauge className="h-4 w-4 text-primary-500" />
                  <span>Quality Metrics</span>
                </h2>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Overall Quality</span>
                    <span className={`badge ${results.quality_metrics.overall_quality === 'good' ? 'badge-success' :
                        sults.quality_metrics.overall_quality === 'medium' ? 'badge-warning' :
                          'bge-danger'
                      }`}>
                      {results.quality_metrics.overall_quality}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Blur Score</span>
                    <span className="font-mono text-xs">{results.quality_metrics.blur_score?.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Brightness</span>
                    <span className="font-mono text-xs">{results.quality_metrics.brightness?.toFixed(1)}</span>
                  </div>
                  {results.quality_metrics.is_blurry && (
                    <div className="p-1.5 bg-warning-50 rounded text-xs text-warning-700">
                      ⚠️ Image appears blurry
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Geolocation Status */}
            {useEnhancedDetection && (
              <div className="card">
                <h2 className="card-header flex items-center space-x-1.5">
                  <MapPin className="h-4 w-4 text-primary-500" />
                  <span>Geolocation</span>
                </h2>

                {geolocation ? (
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Latitude</span>
                      <span className="font-mono text-xs">{geolocation.latitude?.toFixed(6)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Longitude</span>
                      <span className="font-mono text-xs">{geolocation.longitude?.toFixed(6)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Accuracy</span>
                      <span className="font-mono text-xs">±{geolocation.accuracy?.toFixed(0)}m</span>
                    </div>
                    <div className="text-xs text-success-600 flex items-center space-x-1">
                      <span className="w-1.5 h-1.5 bg-success-500 rounded-full"></span>
                      <span>GPS Active</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">
                    {geoError ? (
                      <div className="text-warning-600 text-xs">
                        ⚠️ {geoError}
                      </div>
                    ) : (
                      <div className="text-xs">
                        Location will be captured when detection starts
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Tracking Status */}
            {useEnhancedDetection && results?.tracking_active && (
              <div className="card bg-primary-50 border-primary-200">
                <div className="flex items-center space-x-2 text-primary-800">
                  <RotateCw className="h-4 w-4 animate-spin" />
                  <span className="text-sm font-medium">Object Tracking Active</span>
                </div>
                <p className="text-xs text-primary-600 mt-1">
                  Vehicles are being tracked across frames for consistency
                </p>
              </div>
            )}

            {/* Instructions */}
            <div className="card bg-primary-50 border-primary-200">
              <h3 className="text-sm font-medium text-primary-900 mb-2">Quick Guide</h3>
              <ol className="text-xs text-primary-800 space-y-1 list-decimal list-inside">
                <li>Click "Start Camera" to enable video feed</li>
                <li>Click "Start Detection" to begin scanning</li>
                <li>Position vehicles in frame for detection</li>
                <li>Stolen vehicles will trigger an alert</li>
              </ol>
            </div>
          </div>
        </div>
      </div>

      {/* Stolen Vehicle Alert Modal */}
      <StolenAlert alert={stolenAlert} onClose={clearAlert} />
    </>
  );
};

export default CameraPage;
