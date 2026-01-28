/**
 * Stolen Vehicle Alert Component
 * Displays prominent alert when stolen vehicle is detected
 * 
 * Enhanced with geolocation display
 */

import React from 'react';
import { AlertTriangle, X, Volume2, MapPin, Navigation } from 'lucide-react';

const StolenAlert = ({ alert, onClose }) => {
  if (!alert) return null;

  // Generate Google Maps link for the location
  const getGoogleMapsLink = () => {
    if (alert.geolocation?.latitude && alert.geolocation?.longitude) {
      return `https://www.google.com/maps?q=${alert.geolocation.latitude},${alert.geolocation.longitude}`;
    }
    return null;
  };

  const mapsLink = getGoogleMapsLink();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="alert-stolen rounded-xl shadow-2xl p-6 max-w-md w-full transform scale-100 animate-pulse-fast">
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="h-10 w-10 text-white" />
            <div>
              <h3 className="text-2xl font-bold text-white">STOLEN VEHICLE DETECTED!</h3>
              <p className="text-red-100">Immediate attention required</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-white/80 hover:text-white transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>
        
        <div className="bg-white/20 rounded-lg p-4 mb-4">
          <p className="text-white/80 text-sm mb-2">Detected Plate(s):</p>
          <div className="space-y-2">
            {alert.plates.map((plate, index) => (
              <div
                key={index}
                className="bg-white text-danger-700 font-mono font-bold text-2xl text-center py-2 px-4 rounded-lg"
              >
                {plate}
              </div>
            ))}
          </div>
        </div>

        {/* Geolocation Information */}
        {alert.geolocation && (
          <div className="bg-white/20 rounded-lg p-3 mb-4">
            <div className="flex items-center space-x-2 text-white/90 mb-2">
              <MapPin className="h-4 w-4" />
              <span className="text-sm font-medium">Detection Location</span>
            </div>
            <div className="text-white text-sm font-mono">
              <p>Lat: {alert.geolocation.latitude?.toFixed(6)}</p>
              <p>Lng: {alert.geolocation.longitude?.toFixed(6)}</p>
              {alert.geolocation.accuracy && (
                <p className="text-white/70 text-xs mt-1">
                  Accuracy: ±{alert.geolocation.accuracy?.toFixed(0)}m
                </p>
              )}
            </div>
            {mapsLink && (
              <a
                href={mapsLink}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 inline-flex items-center space-x-1 text-white bg-white/20 hover:bg-white/30 px-3 py-1.5 rounded text-xs font-medium transition-colors"
              >
                <Navigation className="h-3.5 w-3.5" />
                <span>Open in Google Maps</span>
              </a>
            )}
          </div>
        )}
        
        <div className="flex items-center justify-between text-sm text-white/80">
          <span>Time: {new Date(alert.timestamp).toLocaleTimeString()}</span>
          <div className="flex items-center space-x-1">
            <Volume2 className="h-4 w-4" />
            <span>Alert Active</span>
          </div>
        </div>
        
        <button
          onClick={onClose}
          className="mt-4 w-full bg-white text-danger-600 font-bold py-3 rounded-lg hover:bg-gray-100 transition-colors"
        >
          ACKNOWLEDGE ALERT
        </button>
      </div>
    </div>
  );
};

export default StolenAlert;
