/**
 * Video Upload Page Component
 * Upload and analyze video files for vehicle detection
 * With real-time WebSocket streaming for live processing feedback
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';
import { detectionAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { 
  Upload, File, X, Play, CheckCircle, AlertTriangle,
  Clock, Video, Trash2, Download, Wifi, WifiOff, Eye
} from 'lucide-react';

const UploadPage = () => {
  const { token } = useAuth();
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingResults, setProcessingResults] = useState(null);
  const [error, setError] = useState(null);
  
  // Real-time processing state
  const [liveProcessing, setLiveProcessing] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [liveDetections, setLiveDetections] = useState([]);
  const [liveStolenAlerts, setLiveStolenAlerts] = useState([]);
  const [videoInfo, setVideoInfo] = useState(null);
  const [framesProcessed, setFramesProcessed] = useState(0);
  const wsRef = useRef(null);
  
  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/x-matroska', 'video/webm'];
      if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
        setError('Invalid file type. Please upload MP4, AVI, MOV, MKV, or WebM files.');
        return;
      }

      // Validate file size (100MB max)
      if (file.size > 100 * 1024 * 1024) {
        setError('File too large. Maximum size is 100MB.');
        return;
      }

      setSelectedFile(file);
      setError(null);
      setProcessingResults(null);
    }
  }, []);

  // Handle drag and drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const event = { target: { files: [file] } };
      handleFileSelect(event);
    }
  }, [handleFileSelect]);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // Upload video file
  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const response = await detectionAPI.uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      setUploadedFile(response);
      setSelectedFile(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  // WebSocket reconnection settings
  const maxReconnectAttempts = 3;
  const reconnectDelayMs = 2000;
  const reconnectAttemptsRef = useRef(0);
  const isCompletedRef = useRef(false);
  
  // Connect WebSocket with reconnection support
  const connectWebSocket = useCallback((filename, authToken) => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = 'localhost:8000';
    const wsUrl = `${wsProtocol}//${wsHost}/detection/ws/process-video/${filename}?token=${authToken}`;
    
    console.log(`Connecting to WebSocket (attempt ${reconnectAttemptsRef.current + 1}):`, wsUrl);
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
      setError(null);
      reconnectAttemptsRef.current = 0; // Reset on successful connection
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
          case 'video_info':
            setVideoInfo(message.data);
            break;
            
          case 'frame_result':
            const frameData = message.data;
            setCurrentFrame(frameData);
            setProcessingProgress(frameData.progress_percent || 0);
            setFramesProcessed(frameData.frame_number);
            
            if (frameData.detections && frameData.detections.length > 0) {
              setLiveDetections(prev => {
                const newDetections = [...prev, {
                  frame_number: frameData.frame_number,
                  timestamp_ms: frameData.timestamp_ms,
                  detections: frameData.detections
                }];
                return newDetections.slice(-50);
              });
            }
            
            if (frameData.stolen_found && frameData.stolen_found.length > 0) {
              setLiveStolenAlerts(prev => {
                const newAlerts = frameData.stolen_found.filter(
                  plate => !prev.some(a => a.plate === plate)
                ).map(plate => ({
                  plate,
                  frame: frameData.frame_number,
                  timestamp: frameData.timestamp_ms
                }));
                return [...prev, ...newAlerts];
              });
            }
            break;
            
          case 'complete':
            console.log('Processing complete:', message.data);
            isCompletedRef.current = true;
            // Use detections from server (includes all detection details)
            setProcessingResults({
              ...message.data,
              // detections array from backend includes frame_number, timestamp_ms, and detections
              detections: message.data.detections || [],
              unique_plates: message.data.unique_plates || []
            });
            setLiveProcessing(false);
            setIsProcessing(false);
            setWsConnected(false);
            ws.close();
            break;
            
          case 'error':
            console.error('Processing error:', message.message);
            setError(message.message);
            setLiveProcessing(false);
            setIsProcessing(false);
            setWsConnected(false);
            break;
            
          default:
            console.log('Unknown message type:', message.type);
        }
      } catch (e) {
        console.error('Error parsing WS message:', e);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };
    
    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setWsConnected(false);
      
      // Don't reconnect if processing completed or manually closed
      if (isCompletedRef.current || event.code === 1000) {
        return;
      }
      
      // Attempt reconnection if still processing
      if (isProcessing && reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++;
        setError(`Connection lost. Reconnecting (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
        
        setTimeout(() => {
          if (isProcessing && !isCompletedRef.current) {
            connectWebSocket(filename, authToken);
          }
        }, reconnectDelayMs);
      } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
        setError('Connection failed after multiple attempts. Falling back to standard processing...');
        fallbackProcess();
      }
    };
    
    return ws;
  }, [isProcessing, liveDetections, liveStolenAlerts]);

  // Process uploaded video with real-time WebSocket streaming
  const handleProcess = async () => {
    if (!uploadedFile?.filename) return;

    setIsProcessing(true);
    setLiveProcessing(true);
    setError(null);
    setLiveDetections([]);
    setLiveStolenAlerts([]);
    setProcessingProgress(0);
    setFramesProcessed(0);
    setCurrentFrame(null);
    setVideoInfo(null);
    isCompletedRef.current = false;
    reconnectAttemptsRef.current = 0;

    try {
      connectWebSocket(uploadedFile.filename, token);
    } catch (err) {
      console.error('WebSocket setup error:', err);
      setError('Failed to establish real-time connection. Using standard processing...');
      fallbackProcess();
    }
  };
  
  // Fallback to standard processing if WebSocket fails
  const fallbackProcess = async () => {
    if (!uploadedFile?.filename) return;
    
    try {
      const response = await detectionAPI.processVideo(uploadedFile.filename);
      setProcessingResults(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
      setLiveProcessing(false);
    }
  };

  // Delete uploaded video
  const handleDelete = async () => {
    if (!uploadedFile?.filename) return;

    try {
      await detectionAPI.deleteVideo(uploadedFile.filename);
      setUploadedFile(null);
      setProcessingResults(null);
    } catch (err) {
      setError('Failed to delete video.');
    }
  };

  // Reset all state
  const handleReset = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setSelectedFile(null);
    setUploadedFile(null);
    setProcessingResults(null);
    setError(null);
    setUploadProgress(0);
    setLiveProcessing(false);
    setWsConnected(false);
    setProcessingProgress(0);
    setCurrentFrame(null);
    setLiveDetections([]);
    setLiveStolenAlerts([]);
    setVideoInfo(null);
    setFramesProcessed(0);
  };

  return (
    <div className="p-4 lg:p-6">
        <div className="mb-4">
          <h1 className="text-xl font-bold text-gray-900">Video Analysis</h1>
          <p className="text-sm text-gray-600">Upload video files for vehicle and license plate detection</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Upload Section */}
          <div className="space-y-4">
            {/* File Drop Zone */}
            {!uploadedFile && (
              <div className="card">
                <h2 className="card-header">Upload Video</h2>
                
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                    selectedFile 
                      ? 'border-primary-400 bg-primary-50' 
                      : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                  }`}
                >
                  {selectedFile ? (
                    <div className="space-y-3">
                      <File className="h-10 w-10 mx-auto text-primary-500" />
                      <div>
                        <p className="text-sm font-medium text-gray-900">{selectedFile.name}</p>
                        <p className="text-xs text-gray-500">
                          {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                      <button
                        onClick={() => setSelectedFile(null)}
                        className="text-xs text-danger-600 hover:text-danger-700"
                      >
                        Remove
                      </button>
                    </div>
                  ) : (
                    <>
                      <Upload className="h-10 w-10 mx-auto text-gray-400 mb-3" />
                      <p className="text-sm text-gray-600 mb-2">
                        Drag and drop a video file here, or
                      </p>
                      <label className="btn btn-primary cursor-pointer inline-block">
                        <span>Browse Files</span>
                        <input
                          type="file"
                          accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
                          onChange={handleFileSelect}
                          className="hidden"
                        />
                      </label>
                      <p className="text-xs text-gray-400 mt-3">
                        Supported: MP4, AVI, MOV, MKV, WebM (Max 100MB)
                      </p>
                    </>
                  )}
                </div>

                {/* Upload Progress */}
                {isUploading && (
                  <div className="mt-3">
                    <div className="flex justify-between text-xs text-gray-600 mb-1.5">
                      <span>Uploading...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div
                        className="bg-primary-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Upload Button */}
                {selectedFile && !isUploading && (
                  <button
                    onClick={handleUpload}
                    className="mt-3 btn btn-primary w-full flex items-center justify-center space-x-1.5"
                  >
                    <Upload className="h-4 w-4" />
                    <span>Upload Video</span>
                  </button>
                )}
              </div>
            )}

            {/* Uploaded File Info */}
            {uploadedFile && !processingResults && (
              <div className="card">
                <h2 className="card-header">Uploaded Video</h2>
                
                <div className="p-3 bg-success-50 rounded border border-success-200 mb-3">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-success-500" />
                    <div>
                      <p className="text-sm font-medium text-success-800">Upload Complete</p>
                      <p className="text-xs text-success-600">
                        {uploadedFile.video_info?.original_filename}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Video Info */}
                <div className="space-y-1.5 mb-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Resolution</span>
                    <span className="text-gray-900">
                      {uploadedFile.video_info?.width} x {uploadedFile.video_info?.height}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">FPS</span>
                    <span className="text-gray-900">{uploadedFile.video_info?.fps?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Duration</span>
                    <span className="text-gray-900">{uploadedFile.video_info?.duration_seconds}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Total Frames</span>
                    <span className="text-gray-900">{uploadedFile.video_info?.frame_count}</span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-2">
                  <button
                    onClick={handleProcess}
                    disabled={isProcessing}
                    className="btn btn-success flex-1 flex items-center justify-center space-x-1.5"
                  >
                    {isProcessing ? (
                      <>
                        <LoadingSpinner size="sm" />
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4" />
                        <span>Start Analysis</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleDelete}
                    className="btn btn-danger flex items-center space-x-1.5"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-danger-50 border border-danger-200 rounded flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4 text-danger-500 flex-shrink-0" />
                <p className="text-sm text-danger-700">{error}</p>
              </div>
            )}
            
            {/* Live Processing Panel */}
            {liveProcessing && (
              <div className="card border-2 border-primary-300 bg-primary-50/30">
                <h2 className="card-header flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Eye className="h-5 w-5 text-primary-600 animate-pulse" />
                    <span>Live Processing</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    {wsConnected ? (
                      <span className="flex items-center text-xs text-success-600">
                        <Wifi className="h-3 w-3 mr-1" />
                        Connected
                      </span>
                    ) : (
                      <span className="flex items-center text-xs text-gray-400">
                        <WifiOff className="h-3 w-3 mr-1" />
                        Connecting...
                      </span>
                    )}
                  </div>
                </h2>
                
                {/* Progress Bar */}
                <div className="mb-4">
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>Processing Frame {framesProcessed} {videoInfo ? `/ ${videoInfo.total_frames}` : ''}</span>
                    <span>{processingProgress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${processingProgress}%` }}
                    />
                  </div>
                </div>
                
                {/* Current Frame Info */}
                {currentFrame && (
                  <div className="mb-3 p-2 bg-white rounded border text-xs">
                    <div className="flex justify-between text-gray-600">
                      <span>Frame #{currentFrame.frame_number}</span>
                      <span>{(currentFrame.timestamp_ms / 1000).toFixed(2)}s</span>
                    </div>
                    {currentFrame.detections?.length > 0 && (
                      <div className="mt-1 text-primary-700 font-medium">
                        {currentFrame.detections.length} plate(s) detected
                      </div>
                    )}
                  </div>
                )}
                
                {/* Live Stolen Alerts */}
                {liveStolenAlerts.length > 0 && (
                  <div className="mb-3 p-2 bg-danger-100 border border-danger-300 rounded">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="h-4 w-4 text-danger-600" />
                      <span className="text-sm font-bold text-danger-800">
                        Stolen Vehicle Alert!
                      </span>
                    </div>
                    <div className="space-y-1">
                      {liveStolenAlerts.map((alert, idx) => (
                        <div key={idx} className="bg-white p-2 rounded text-center">
                          <span className="font-mono font-bold text-danger-700">
                            {alert.plate}
                          </span>
                          <span className="text-xs text-gray-500 ml-2">
                            Frame {alert.frame}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Live Detection Feed */}
                <div className="max-h-48 overflow-y-auto">
                  <h4 className="text-xs font-medium text-gray-500 mb-2">Recent Detections</h4>
                  {liveDetections.length > 0 ? (
                    <div className="space-y-1">
                      {[...liveDetections].reverse().slice(0, 10).map((frame, idx) => (
                        <div key={idx} className="p-2 bg-white rounded border text-xs">
                          <div className="flex justify-between text-gray-500 mb-1">
                            <span>Frame {frame.frame_number}</span>
                            <span>{(frame.timestamp_ms / 1000).toFixed(2)}s</span>
                          </div>
                          {frame.detections.map((det, detIdx) => (
                            <div 
                              key={detIdx} 
                              className={`px-2 py-1 rounded ${
                                det.is_stolen 
                                  ? 'bg-danger-100 text-danger-700' 
                                  : 'bg-gray-100 text-gray-700'
                              }`}
                            >
                              <span className="font-mono font-medium">
                                {det.plate_text || 'Unknown'}
                              </span>
                              <span className="ml-2 text-gray-500">
                                {det.confidence ? `${Math.round(det.confidence * 100)}%` : ''}
                              </span>
                              {det.is_stolen && (
                                <span className="ml-2 text-danger-600 font-bold">STOLEN</span>
                              )}
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4 text-gray-400 text-xs">
                      Waiting for detections...
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-4">
            {processingResults ? (
              <>
                {/* Summary Card */}
                <div className="card">
                  <h2 className="card-header flex items-center justify-between">
                    <span>Analysis Results</span>
                    <button
                      onClick={handleReset}
                      className="text-xs text-primary-600 hover:text-primary-700"
                    >
                      New Upload
                    </button>
                  </h2>

                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="p-3 bg-primary-50 rounded text-center">
                      <p className="text-2xl font-bold text-primary-700">
                        {processingResults.frames_with_detections}
                      </p>
                      <p className="text-xs text-primary-600">Frames with Detections</p>
                    </div>
                    <div className={`p-3 rounded text-center ${
                      processingResults.stolen_vehicles_found?.length > 0 
                        ? 'bg-danger-50' 
                        : 'bg-success-50'
                    }`}>
                      <p className={`text-2xl font-bold ${
                        processingResults.stolen_vehicles_found?.length > 0 
                          ? 'text-danger-700' 
                          : 'text-success-700'
                      }`}>
                        {processingResults.stolen_vehicles_found?.length || 0}
                      </p>
                      <p className={`text-xs ${
                        processingResults.stolen_vehicles_found?.length > 0 
                          ? 'text-danger-600' 
                          : 'text-success-600'
                      }`}>
                        Stolen Vehicles Found
                      </p>
                    </div>
                  </div>

                  <div className="space-y-1.5 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Processing Time</span>
                      <span className="text-gray-900">
                        {(processingResults.total_processing_time_ms / 1000).toFixed(2)}s
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Frames Analyzed</span>
                      <span className="text-gray-900">{processingResults.total_frames_processed}</span>
                    </div>
                  </div>
                </div>

                {/* Stolen Vehicles Alert */}
                {processingResults.stolen_vehicles_found?.length > 0 && (
                  <div className="card bg-danger-50 border-danger-200">
                    <h3 className="text-sm font-bold text-danger-800 flex items-center space-x-2 mb-3">
                      <AlertTriangle className="h-4 w-4" />
                      <span>Stolen Vehicles Detected!</span>
                    </h3>
                    <div className="space-y-2">
                      {processingResults.stolen_vehicles_found.map((plate, index) => (
                        <div key={index} className="bg-white p-2.5 rounded font-mono text-sm font-bold text-danger-700 text-center">
                          {plate}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Unique Plates Summary */}
                {processingResults.unique_plates?.length > 0 && (
                  <div className="card">
                    <h3 className="card-header">Plates Detected</h3>
                    <div className="space-y-2">
                      {processingResults.unique_plates.map((plate, index) => (
                        <div 
                          key={index} 
                          className={`p-3 rounded border ${
                            plate.is_stolen 
                              ? 'bg-danger-50 border-danger-200' 
                              : 'bg-gray-50 border-gray-200'
                          }`}
                        >
                          <div className="flex justify-between items-center">
                            <span className={`font-mono text-lg font-bold ${
                              plate.is_stolen ? 'text-danger-700' : 'text-gray-900'
                            }`}>
                              {plate.plate_text}
                            </span>
                            {plate.is_stolen && (
                              <span className="px-2 py-1 bg-danger-100 text-danger-700 text-xs font-bold rounded">
                                STOLEN
                              </span>
                            )}
                          </div>
                          <div className="mt-1 text-xs text-gray-500 flex justify-between">
                            <span>Frame {plate.frame_number} • {(plate.timestamp_ms / 1000).toFixed(2)}s</span>
                            <span>{Math.round(plate.confidence * 100)}% confidence</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Detection Details */}
                <div className="card max-h-80 overflow-y-auto">
                  <h3 className="card-header sticky top-0 bg-white">Detection Timeline</h3>
                  
                  <div className="space-y-2">
                    {processingResults.detections?.length > 0 ? (
                      processingResults.detections.map((frame, index) => (
                      <div key={index} className="p-2.5 bg-gray-50 rounded">
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="text-xs font-medium text-gray-700">
                            Frame {frame.frame_number}
                          </span>
                          <span className="text-xs text-gray-500">
                            {(frame.timestamp_ms / 1000).toFixed(2)}s
                          </span>
                        </div>
                        
                        {frame.detections?.map((det, detIndex) => (
                          <div
                            key={detIndex}
                            className={`mt-1.5 p-2 rounded border ${
                              det.is_stolen 
                                ? 'bg-danger-50 border-danger-200' 
                                : 'bg-white border-gray-200'
                            }`}
                          >
                            <div className="flex justify-between items-center">
                              <span className={`font-mono text-sm font-medium ${
                                det.is_stolen ? 'text-danger-700' : 'text-gray-900'
                              }`}>
                                {det.plate_text || 'Unknown'}
                              </span>
                              {det.is_stolen && (
                                <span className="badge badge-danger">STOLEN</span>
                              )}
                            </div>
                            <p className="text-xs text-gray-500 mt-0.5">
                              {det.vehicle_type || 'vehicle'} • {det.confidence ? `${Math.round(det.confidence * 100)}%` : '80%'} confidence
                            </p>
                          </div>
                        ))}
                      </div>
                    ))
                    ) : (
                      <div className="text-center py-6 text-gray-400">
                        <Video className="h-8 w-8 mx-auto mb-2" />
                        <p className="text-sm">No plates detected in this video</p>
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="card">
                <div className="text-center py-8 text-gray-400">
                  <Video className="h-12 w-12 mx-auto mb-3" />
                  <p className="text-sm font-medium">No results yet</p>
                  <p className="text-xs">Upload and process a video to see detection results</p>
                </div>
              </div>
            )}
          </div>
        </div>
    </div>
  );
};

export default UploadPage;
