/**
 * API Service Module
 * Handles all HTTP requests to the backend API
 */

import axios from 'axios';

// API base URL - adjust for production
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid - clear auth data
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ============ Auth API ============

export const authAPI = {
  /**
   * Login user and get JWT token
   */
  login: async (username, password) => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },

  /**
   * Register new user (admin only)
   */
  register: async (username, password, role = 'user') => {
    const response = await api.post('/auth/register', { username, password, role });
    return response.data;
  },

  /**
   * Get current user profile
   */
  getProfile: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  /**
   * Initialize admin account (first-time setup)
   */
  initAdmin: async () => {
    const response = await api.post('/auth/init-admin');
    return response.data;
  },

  /**
   * Change password
   */
  changePassword: async (oldPassword, newPassword) => {
    const response = await api.post('/auth/change-password', null, {
      params: { old_password: oldPassword, new_password: newPassword }
    });
    return response.data;
  },

  /**
   * Get all users (admin only)
   */
  getUsers: async () => {
    const response = await api.get('/auth/users');
    return response.data;
  },

  /**
   * Delete user (admin only)
   */
  deleteUser: async (userId) => {
    const response = await api.delete(`/auth/users/${userId}`);
    return response.data;
  }
};

// ============ Stolen Vehicles API ============

export const stolenVehiclesAPI = {
  /**
   * Get all stolen vehicles
   */
  getAll: async (activeOnly = true) => {
    const response = await api.get('/stolen-vehicles/', {
      params: { active_only: activeOnly }
    });
    return response.data;
  },

  /**
   * Get single stolen vehicle by ID
   */
  getById: async (id) => {
    const response = await api.get(`/stolen-vehicles/${id}`);
    return response.data;
  },

  /**
   * Register new stolen vehicle
   */
  create: async (plateNumber, vehicleType, description = null) => {
    const response = await api.post('/stolen-vehicles/', {
      plate_number: plateNumber,
      vehicle_type: vehicleType,
      description
    });
    return response.data;
  },

  /**
   * Update stolen vehicle
   */
  update: async (id, data) => {
    const response = await api.put(`/stolen-vehicles/${id}`, data);
    return response.data;
  },

  /**
   * Delete stolen vehicle
   */
  delete: async (id) => {
    const response = await api.delete(`/stolen-vehicles/${id}`);
    return response.data;
  },

  /**
   * Mark vehicle as resolved
   */
  markResolved: async (id) => {
    const response = await api.post(`/stolen-vehicles/${id}/resolve`);
    return response.data;
  },

  /**
   * Check if plate is stolen
   */
  checkPlate: async (plateNumber) => {
    const response = await api.get(`/stolen-vehicles/check/${plateNumber}`);
    return response.data;
  }
};

// ============ Detection API ============

export const detectionAPI = {
  /**
   * Process single frame/image (base64 in URL params - small images)
   */
  detectFrame: async (imageBase64, returnAnnotated = true) => {
    const response = await api.post('/detection/frame', null, {
      params: { 
        image_data: imageBase64,
        return_annotated: returnAnnotated
      }
    });
    return response.data;
  },

  /**
   * Process frame with image in body (for larger images)
   */
  detectFramePost: async (imageBase64, returnAnnotated = true) => {
    const response = await api.post('/detection/frame-upload', {
      image_data: imageBase64,
      return_annotated: returnAnnotated
    });
    return response.data;
  },

  /**
   * Enhanced detection with geolocation and advanced features
   * - Object tracking
   * - Quality enhancement
   * - Plate format validation
   * - Geolocation capture
   */
  detectFrameEnhanced: async (imageBase64, returnAnnotated = true, latitude = null, longitude = null, accuracy = null, locationName = null) => {
    const response = await api.post('/detection/frame-enhanced', {
      image_data: imageBase64,
      return_annotated: returnAnnotated,
      latitude: latitude,
      longitude: longitude,
      location_accuracy: accuracy,
      location_name: locationName
    });
    return response.data;
  },

  /**
   * Check frame quality without full detection
   */
  checkQuality: async (imageBase64) => {
    const response = await api.get('/detection/quality-check', {
      params: { image_data: imageBase64 }
    });
    return response.data;
  },

  /**
   * Reset object tracking state
   */
  resetTracking: async () => {
    const response = await api.post('/detection/reset-tracking');
    return response.data;
  },

  /**
   * Upload video file
   */
  uploadVideo: async (file, onProgress = null) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/detection/upload-video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress ? (e) => {
        const percent = Math.round((e.loaded * 100) / e.total);
        onProgress(percent);
      } : undefined
    });
    return response.data;
  },

  /**
   * Process uploaded video
   */
  processVideo: async (filename, frameSkip = null) => {
    const params = {};
    if (frameSkip) params.frame_skip = frameSkip;
    
    const response = await api.post(`/detection/process-video/${filename}`, null, {
      params
    });
    return response.data;
  },

  /**
   * Delete uploaded video
   */
  deleteVideo: async (filename) => {
    const response = await api.delete(`/detection/video/${filename}`);
    return response.data;
  }
};

// ============ Detection Logs API ============

export const logsAPI = {
  /**
   * Get detection logs with pagination
   */
  getAll: async (limit = 100, offset = 0, stolenOnly = false) => {
    const response = await api.get('/logs/', {
      params: { limit, offset, stolen_only: stolenOnly }
    });
    return response.data;
  },

  /**
   * Get detection statistics
   */
  getStatistics: async () => {
    const response = await api.get('/logs/statistics');
    return response.data;
  },

  /**
   * Get recent stolen detections
   */
  getRecentStolen: async (limit = 10) => {
    const response = await api.get('/logs/recent-stolen', {
      params: { limit }
    });
    return response.data;
  },

  /**
   * Get today's logs
   */
  getToday: async () => {
    const response = await api.get('/logs/today');
    return response.data;
  },

  /**
   * Get logs by date range
   */
  getByDateRange: async (startDate, endDate) => {
    const response = await api.get('/logs/by-date', {
      params: { start_date: startDate, end_date: endDate }
    });
    return response.data;
  }
};

// ============ Health API ============

export const healthAPI = {
  /**
   * Check system health
   */
  check: async () => {
    const response = await api.get('/health');
    return response.data;
  }
};

export default api;
