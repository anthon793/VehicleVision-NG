/**
 * Test Detection Page
 * Upload a single image to test the detection model
 */

import React, { useState, useRef, useCallback } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';
import StolenAlert from '../components/StolenAlert';
import { detectionAPI } from '../services/api';
import { 
  Image, Upload, Camera, CheckCircle, AlertTriangle, 
  X, RefreshCw, Car, FileText, ZoomIn
} from 'lucide-react';

const TestDetectionPage = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [error, setError] = useState(null);
  const [stolenAlert, setStolenAlert] = useState(null);
  
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  // Handle file selection
  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file (JPG, PNG, etc.)');
        return;
      }

      // Validate file size (10MB max)
      if (file.size > 10 * 1024 * 1024) {
        setError('File too large. Maximum size is 10MB.');
        return;
      }

      setSelectedImage(file);
      setError(null);
      setResults(null);
      setAnnotatedImage(null);
      setStolenAlert(null);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
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

  // Process image
  const handleProcess = async () => {
    if (!imagePreview) return;

    setIsProcessing(true);
    setError(null);
    setResults(null);
    setAnnotatedImage(null);
    setStolenAlert(null);

    try {
      // Send image for detection
      const response = await detectionAPI.detectFramePost(imagePreview, true);
      
      setResults(response);
      
      // Set annotated image if available
      if (response.annotated_image) {
        setAnnotatedImage(`data:image/jpeg;base64,${response.annotated_image}`);
      }
      
      // Check for stolen vehicles
      if (response.has_stolen) {
        const stolenPlate = response.plates?.find(p => p.is_stolen);
        if (stolenPlate) {
          setStolenAlert({
            plateNumber: stolenPlate.plate_text,
            vehicleType: stolenPlate.vehicle_type || 'Unknown'
          });
        }
      }
    } catch (err) {
      console.error('Detection error:', err);
      setError(err.response?.data?.detail || 'Detection failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset all state
  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setAnnotatedImage(null);
    setError(null);
    setStolenAlert(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <>
      {/* Stolen Alert */}
      {stolenAlert && (
        <StolenAlert
          plateNumber={stolenAlert.plateNumber}
          vehicleType={stolenAlert.vehicleType}
          onDismiss={() => setStolenAlert(null)}
        />
      )}
      
      <div className="p-6 lg:p-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Test Detection Model</h1>
          <p className="text-gray-600">Upload an image to test vehicle and license plate detection</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="card">
              <h2 className="card-header flex items-center space-x-2">
                <Image className="h-5 w-5 text-primary-600" />
                <span>Upload Image</span>
              </h2>
              
              {/* File Drop Zone */}
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                  imagePreview 
                    ? 'border-primary-400 bg-primary-50' 
                    : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                
                {imagePreview ? (
                  <div className="space-y-4">
                    <img 
                      src={imagePreview} 
                      alt="Preview" 
                      className="max-h-64 mx-auto rounded-lg shadow-md"
                    />
                    <p className="text-sm text-gray-600">{selectedImage?.name}</p>
                    <p className="text-xs text-gray-500">Click to select a different image</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full">
                      <Upload className="h-8 w-8 text-primary-600" />
                    </div>
                    <div>
                      <p className="text-gray-600">
                        <span className="text-primary-600 font-medium">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-sm text-gray-500 mt-1">JPG, PNG, WEBP up to 10MB</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Error Message */}
              {error && (
                <div className="mt-4 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center space-x-3">
                  <AlertTriangle className="h-5 w-5 text-danger-500 flex-shrink-0" />
                  <p className="text-sm text-danger-700">{error}</p>
                  <button onClick={() => setError(null)} className="ml-auto">
                    <X className="h-4 w-4 text-danger-500" />
                  </button>
                </div>
              )}

              {/* Action Buttons */}
              <div className="mt-6 flex space-x-4">
                <button
                  onClick={handleProcess}
                  disabled={!imagePreview || isProcessing}
                  className="btn-primary flex-1 flex items-center justify-center space-x-2"
                >
                  {isProcessing ? (
                    <>
                      <LoadingSpinner size="sm" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <ZoomIn className="h-5 w-5" />
                      <span>Detect Vehicles & Plates</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={handleReset}
                  disabled={isProcessing}
                  className="btn-secondary flex items-center space-x-2"
                >
                  <RefreshCw className="h-5 w-5" />
                  <span>Reset</span>
                </button>
              </div>
            </div>

            {/* Instructions */}
            <div className="card bg-blue-50 border-blue-200">
              <h3 className="font-semibold text-blue-900 mb-2">How to Test</h3>
              <ol className="text-sm text-blue-800 space-y-2 list-decimal list-inside">
                <li>Upload an image containing a vehicle with a visible license plate</li>
                <li>Click "Detect Vehicles & Plates" to run the detection</li>
                <li>View the results showing detected vehicles and extracted plate numbers</li>
                <li>If a plate matches the stolen database, an alert will appear</li>
              </ol>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Annotated Image */}
            <div className="card">
              <h2 className="card-header flex items-center space-x-2">
                <Camera className="h-5 w-5 text-primary-600" />
                <span>Detection Results</span>
              </h2>
              
              {annotatedImage ? (
                <div className="space-y-4">
                  <img 
                    src={annotatedImage} 
                    alt="Annotated Result" 
                    className="w-full rounded-lg shadow-md"
                  />
                  <p className="text-sm text-gray-500 text-center">
                    Detected vehicles and plates are highlighted
                  </p>
                </div>
              ) : (
                <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                  <div className="text-center text-gray-500">
                    <Image className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>Detection results will appear here</p>
                  </div>
                </div>
              )}
            </div>

            {/* Detection Details */}
            {results && (
              <div className="card">
                <h2 className="card-header flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-primary-600" />
                  <span>Detection Details</span>
                </h2>
                
                <div className="space-y-4">
                  {/* Summary */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4 text-center">
                      <Car className="h-8 w-8 mx-auto mb-2 text-primary-600" />
                      <p className="text-2xl font-bold text-gray-900">
                        {results.plates?.length || 0}
                      </p>
                      <p className="text-sm text-gray-600">Plates Detected</p>
                    </div>
                    <div className={`rounded-lg p-4 text-center ${
                      results.has_stolen ? 'bg-danger-50' : 'bg-success-50'
                    }`}>
                      {results.has_stolen ? (
                        <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-danger-600" />
                      ) : (
                        <CheckCircle className="h-8 w-8 mx-auto mb-2 text-success-600" />
                      )}
                      <p className={`text-lg font-bold ${
                        results.has_stolen ? 'text-danger-700' : 'text-success-700'
                      }`}>
                        {results.has_stolen ? 'STOLEN FOUND!' : 'All Clear'}
                      </p>
                      <p className={`text-sm ${
                        results.has_stolen ? 'text-danger-600' : 'text-success-600'
                      }`}>
                        {results.has_stolen ? 'Match in database' : 'No matches'}
                      </p>
                    </div>
                  </div>

                  {/* Processing Time */}
                  <div className="text-sm text-gray-500 text-center">
                    Processing time: {(results.processing_time_ms || 0).toFixed(0)}ms
                  </div>

                  {/* Detected Plates List */}
                  {results.plates && results.plates.length > 0 ? (
                    <div className="space-y-3">
                      <h3 className="font-medium text-gray-900">Detected Plates:</h3>
                      {results.plates.map((plate, index) => (
                        <div 
                          key={index}
                          className={`p-4 rounded-lg border ${
                            plate.is_stolen 
                              ? 'bg-danger-50 border-danger-300' 
                              : 'bg-gray-50 border-gray-200'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className={`text-lg font-mono font-bold ${
                                plate.is_stolen ? 'text-danger-700' : 'text-gray-900'
                              }`}>
                                {plate.plate_text || 'Unable to read'}
                              </p>
                              <p className="text-sm text-gray-600">
                                Vehicle: {plate.vehicle_type || 'Unknown'} | 
                                Confidence: {((plate.confidence || 0) * 100).toFixed(1)}%
                              </p>
                            </div>
                            {plate.is_stolen && (
                              <span className="px-3 py-1 bg-danger-600 text-white text-sm font-medium rounded-full">
                                STOLEN
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 py-4">
                      <p>No license plates detected in this image</p>
                      <p className="text-sm">Try an image with a clearer view of the vehicle's plate</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default TestDetectionPage;
