/**
 * Detection Logs Page Component
 * View detection history with filtering options
 * 
 * Enhanced with geolocation display
 */

import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';
import { logsAPI } from '../services/api';
import {
  History, Filter, Search, AlertTriangle, CheckCircle,
  Calendar, RefreshCw, Download, ChevronLeft, ChevronRight,
  MapPin, Shield, Gauge, ChevronDown, ChevronUp
} from 'lucide-react';

const LogsPage = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all'); // all, stolen, not_stolen
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [expandedLog, setExpandedLog] = useState(null);

  const PAGE_SIZE = 50;

  // Fetch logs
  const fetchLogs = async (reset = false) => {
    setLoading(true);
    setError(null);

    try {
      const offset = reset ? 0 : (page - 1) * PAGE_SIZE;
      const stolenOnly = filter === 'stolen';

      const data = await logsAPI.getAll(PAGE_SIZE, offset, stolenOnly);

      if (reset) {
        setLogs(data);
      } else {
        setLogs(prev => [...prev, ...data]);
      }

      setHasMore(data.length === PAGE_SIZE);
    } catch (err) {
      setError('Failed to load detection logs');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Initial load and filter changes
  useEffect(() => {
    setPage(1);
    fetchLogs(true);
  }, [filter]);

  // Filtered logs based on search
  const filteredLogs = logs.filter(log => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      log.detected_plate?.toLowerCase().includes(query) ||
      log.vehicle_type?.toLowerCase().includes(query) ||
      log.source_type?.toLowerCase().includes(query)
    );
  });

  // Apply additional filter for not_stolen
  const displayLogs = filter === 'not_stolen'
    ? filteredLogs.filter(log => log.match_status === 'not_stolen')
    : filteredLogs;

  // Export logs as CSV
  const exportCSV = () => {
    const headers = ['ID', 'Plate', 'Vehicle Type', 'Status', 'Confidence', 'Source', 'Latitude', 'Longitude', 'Location', 'Validated', 'Quality', 'Timestamp'];
    const rows = displayLogs.map(log => [
      log.id,
      log.detected_plate,
      log.vehicle_type || '',
      log.match_status,
      log.confidence_score?.toFixed(2) || '',
      log.source_type || '',
      log.latitude || '',
      log.longitude || '',
      log.location_name || '',
      log.is_validated ? 'Yes' : 'No',
      log.quality_score?.toFixed(2) || '',
      new Date(log.timestamp).toISOString()
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `detection_logs_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  // Toggle expanded log details
  const toggleExpand = (logId) => {
    setExpandedLog(expandedLog === logId ? null : logId);
  };

  return (
    <div className="p-4 lg:p-6">
      <div className="mb-4">
        <h1 className="text-xl font-bold text-gray-900">Detection Logs</h1>
        <p className="text-sm text-gray-600">View and search through all vehicle detections</p>
      </div>

      {/* Filters and Actions */}
      <div className="card mb-4">
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Search by plate, vehicle type, or source..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          {/* Filter Buttons */}
          <div className="flex items-center space-x-1.5">
            <Filter className="h-4 w-4 text-gray-400" />
            <button
              onClick={() => setFilter('all')}
              className={`px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${filter === 'all'
                  ? 'bg-primary-100 text-primary-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('stolen')}
              className={`px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${filter === 'stolen'
                  ? 'bg-danger-100 text-danger-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              Stolen
            </button>
            <button
              onClick={() => setFilter('not_stolen')}
              className={`px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${filter === 'not_stolen'
                  ? 'bg-success-100 text-success-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              Clear
            </button>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-1.5">
            <button
              onClick={() => fetchLogs(true)}
              className="btn btn-secondary flex items-center space-x-1.5"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Refresh</span>
            </button>
            <button
              onClick={exportCSV}
              className="btn btn-secondary flex items-center space-x-1.5"
            >
              <Download className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Export</span>
            </button>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-3 bg-danger-50 border border-danger-200 rounded text-sm text-danger-700">
          {error}
        </div>
      )}

      {/* Logs Table */}
      <div className="card p-0 overflow-hidden">
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th>Plate Number</th>
                <th>Vehicle Type</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {loading && logs.length === 0 ? (
                <tr>
                  <td colSpan="6" className="text-center py-8">
                    <LoadingSpinner text="Loading logs..." />
                  </td>
                </tr>
              ) : displayLogs.length === 0 ? (
                <tr>
                  <td colSpan="6" className="text-center py-8 text-gray-400">
                    <History className="h-8 w-8 mx-auto mb-2" />
                    <p className="text-sm">No detection logs found</p>
                  </td>
                </tr>
              ) : (
                displayLogs.map((log) => (
                  <React.Fragment key={log.id}>
                    <tr
                      className="cursor-pointer hover:bg-gray-50"
                      onClick={() => toggleExpand(log.id)}
                    >
                      <td>
                        <div className="flex items-center space-x-2">
                          <span className="font-mono font-semibold text-gray-900">
                            {log.detected_plate}
                          </span>
                          {log.is_validated && (
                            <Shield className="h-3.5 w-3.5 text-success-500" title="Validated format" />
                          )}
                          {log.latitude && log.longitude && (
                            <MapPin className="h-3.5 w-3.5 text-primary-500" title="Has location" />
                          )}
                        </div>
                      </td>
                      <td>
                        <span className="text-gray-600 capitalize">
                          {log.vehicle_type || 'Unknown'}
                        </span>
                      </td>
                      <td>
                        {log.match_status === 'stolen' ? (
                          <span className="badge badge-danger flex items-center space-x-1 w-fit">
                            <AlertTriangle className="h-3 w-3" />
                            <span>Stolen</span>
                          </span>
                        ) : log.match_status === 'not_stolen' ? (
                          <span className="badge badge-success flex items-center space-x-1 w-fit">
                            <CheckCircle className="h-3 w-3" />
                            <span>Clear</span>
                          </span>
                        ) : (
                          <span className="badge badge-warning">Unknown</span>
                        )}
                      </td>
                      <td>
                        <span className="text-gray-600">
                          {log.confidence_score ? `${(log.confidence_score * 100).toFixed(0)}%` : '-'}
                        </span>
                      </td>
                      <td>
                        <span className="badge badge-info capitalize">
                          {log.source_type || 'N/A'}
                        </span>
                      </td>
                      <td>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-gray-900">
                              {new Date(log.timestamp).toLocaleDateString()}
                            </p>
                            <p className="text-xs text-gray-400">
                              {new Date(log.timestamp).toLocaleTimeString()}
                            </p>
                          </div>
                          {expandedLog === log.id ? (
                            <ChevronUp className="h-4 w-4 text-gray-400" />
                          ) : (
                            <ChevronDown className="h-4 w-4 text-gray-400" />
                          )}
                        </div>
                      </td>
                    </tr>
                    {/* Expanded Details Row */}
                    {expandedLog === log.id && (
                      <tr className="bg-gray-50">
                        <td colSpan="6" className="px-4 py-3">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                            {/* Geolocation */}
                            <div>
                              <p className="text-gray-500 font-medium mb-1 flex items-center">
                                <MapPin className="h-3 w-3 mr-1" />
                                Location
                              </p>
                              {log.latitude && log.longitude ? (
                                <div>
                                  <p className="font-mono">{log.latitude?.toFixed(6)}, {log.longitude?.toFixed(6)}</p>
                                  {log.location_name && (
                                    <p className="text-gray-500">{log.location_name}</p>
                                  )}
                                  {log.location_accuracy && (
                                    <p className="text-gray-400">±{log.location_accuracy?.toFixed(0)}m accuracy</p>
                                  )}
                                </div>
                              ) : (
                                <p className="text-gray-400">Not available</p>
                              )}
                            </div>

                            {/* Quality */}
                            <div>
                              <p className="text-gray-500 font-medium mb-1 flex items-center">
                                <Gauge className="h-3 w-3 mr-1" />
                                Quality
                              </p>
                              {log.quality_score ? (
                                <div>
                                  <p>Score: {(log.quality_score * 100).toFixed(0)}%</p>
                                  <p className="text-gray-500">Level: {log.processing_level || 'standard'}</p>
                                </div>
                              ) : (
                                <p className="text-gray-400">Not assessed</p>
                              )}
                            </div>

                            {/* Validation */}
                            <div>
                              <p className="text-gray-500 font-medium mb-1 flex items-center">
                                <Shield className="h-3 w-3 mr-1" />
                                Validation
                              </p>
                              <p className={log.is_validated ? 'text-success-600' : 'text-gray-400'}>
                                {log.is_validated ? 'Valid plate format' : 'Not validated'}
                              </p>
                            </div>

                            {/* Additional Info */}
                            <div>
                              <p className="text-gray-500 font-medium mb-1">Details</p>
                              <div>
                                {log.track_id && <p>Track ID: {log.track_id}</p>}
                                {log.ocr_confidence && <p>OCR: {(log.ocr_confidence * 100).toFixed(0)}%</p>}
                                {log.processing_time_ms && <p>Time: {log.processing_time_ms.toFixed(0)}ms</p>}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Load More */}
        {hasMore && displayLogs.length > 0 && (
          <div className="p-3 border-t border-gray-200 text-center">
            <button
              onClick={() => {
                setPage(prev => prev + 1);
                fetchLogs(false);
              }}
              disabled={loading}
              className="btn btn-secondary"
            >
              {loading ? 'Loading...' : 'Load More'}
            </button>
          </div>
        )}
      </div>

      {/* Summary */}
      <div className="mt-3 text-xs text-gray-500 text-center">
        Showing {displayLogs.length} of {logs.length} logs
        {searchQuery && ` (filtered by "${searchQuery}")`}
      </div>
    </div>
  );
};

export default LogsPage;
