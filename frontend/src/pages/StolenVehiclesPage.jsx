/**
 * Stolen Vehicles Management Page (Admin Only)
 * CRUD operations for stolen vehicle database
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import LoadingSpinner from '../components/LoadingSpinner';
import { stolenVehiclesAPI } from '../services/api';
import { 
  Car, Plus, Trash2, Edit, CheckCircle, X, Search,
  AlertTriangle, Calendar, RefreshCw
} from 'lucide-react';

const StolenVehiclesPage = () => {
  const { isAdmin } = useAuth();
  const navigate = useNavigate();
  
  const [vehicles, setVehicles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingVehicle, setEditingVehicle] = useState(null);
  const [formData, setFormData] = useState({
    plate_number: '',
    vehicle_type: 'car',
    description: ''
  });
  const [submitting, setSubmitting] = useState(false);

  // Redirect non-admin users
  useEffect(() => {
    if (!isAdmin()) {
      navigate('/dashboard');
    }
  }, [isAdmin, navigate]);

  // Fetch vehicles
  const fetchVehicles = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await stolenVehiclesAPI.getAll(false);
      setVehicles(data);
    } catch (err) {
      setError('Failed to load stolen vehicles');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVehicles();
  }, []);

  // Filter vehicles
  const filteredVehicles = vehicles.filter(v => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      v.plate_number.toLowerCase().includes(query) ||
      v.vehicle_type.toLowerCase().includes(query) ||
      v.description?.toLowerCase().includes(query)
    );
  });

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      if (editingVehicle) {
        await stolenVehiclesAPI.update(editingVehicle.id, formData);
      } else {
        await stolenVehiclesAPI.create(
          formData.plate_number,
          formData.vehicle_type,
          formData.description || null
        );
      }
      
      setShowForm(false);
      setEditingVehicle(null);
      setFormData({ plate_number: '', vehicle_type: 'car', description: '' });
      fetchVehicles();
    } catch (err) {
      setError(err.response?.data?.detail || 'Operation failed');
    } finally {
      setSubmitting(false);
    }
  };

  // Handle delete
  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this record?')) return;

    try {
      await stolenVehiclesAPI.delete(id);
      fetchVehicles();
    } catch (err) {
      setError('Failed to delete vehicle');
    }
  };

  // Handle mark resolved
  const handleResolve = async (id) => {
    try {
      await stolenVehiclesAPI.markResolved(id);
      fetchVehicles();
    } catch (err) {
      setError('Failed to mark vehicle as resolved');
    }
  };

  // Open edit form
  const handleEdit = (vehicle) => {
    setEditingVehicle(vehicle);
    setFormData({
      plate_number: vehicle.plate_number,
      vehicle_type: vehicle.vehicle_type,
      description: vehicle.description || ''
    });
    setShowForm(true);
  };

  return (
    <div className="p-4 lg:p-6">
        <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Stolen Vehicles Database</h1>
            <p className="text-sm text-gray-600">Manage reported stolen vehicle records</p>
          </div>
          
          <button
            onClick={() => {
              setShowForm(true);
              setEditingVehicle(null);
              setFormData({ plate_number: '', vehicle_type: 'car', description: '' });
            }}
            className="btn btn-primary flex items-center space-x-1.5"
          >
            <Plus className="h-4 w-4" />
            <span>Add Vehicle</span>
          </button>
        </div>

        {/* Search and Filters */}
        <div className="card mb-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
              <input
                type="text"
                placeholder="Search by plate number, type, or description..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            <button
              onClick={fetchVehicles}
              className="btn btn-secondary flex items-center space-x-1.5"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-danger-50 border border-danger-200 rounded text-sm text-danger-700 flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4" />
            <span>{error}</span>
            <button onClick={() => setError(null)} className="ml-auto">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Add/Edit Form Modal */}
        {showForm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-5">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-gray-900">
                  {editingVehicle ? 'Edit Vehicle' : 'Add Stolen Vehicle'}
                </h2>
                <button
                  onClick={() => {
                    setShowForm(false);
                    setEditingVehicle(null);
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <form onSubmit={handleSubmit} className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Plate Number *
                  </label>
                  <input
                    type="text"
                    value={formData.plate_number}
                    onChange={(e) => setFormData({ ...formData, plate_number: e.target.value.toUpperCase() })}
                    className="input font-mono"
                    placeholder="LAG 234 ABC"
                    required
                    maxLength={20}
                  />
                  <p className="text-xs text-gray-400 mt-0.5">Will be normalized (uppercase, no spaces)</p>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Vehicle Type *
                  </label>
                  <select
                    value={formData.vehicle_type}
                    onChange={(e) => setFormData({ ...formData, vehicle_type: e.target.value })}
                    className="input"
                    required
                  >
                    <option value="car">Car</option>
                    <option value="motorcycle">Motorcycle</option>
                    <option value="truck">Truck</option>
                    <option value="bus">Bus</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="input"
                    rows={2}
                    placeholder="Vehicle color, model, circumstances..."
                    maxLength={500}
                  />
                </div>

                <div className="flex space-x-2 pt-3">
                  <button
                    type="button"
                    onClick={() => {
                      setShowForm(false);
                      setEditingVehicle(null);
                    }}
                    className="btn btn-secondary flex-1"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="btn btn-primary flex-1 flex items-center justify-center space-x-1.5"
                  >
                    {submitting ? (
                      <>
                        <LoadingSpinner size="sm" />
                        <span>Saving...</span>
                      </>
                    ) : (
                      <span>{editingVehicle ? 'Update' : 'Add Vehicle'}</span>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Vehicles Table */}
        <div className="card p-0 overflow-hidden">
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Plate Number</th>
                  <th>Vehicle Type</th>
                  <th>Status</th>
                  <th>Date Reported</th>
                  <th>Description</th>
                  <th className="text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {loading ? (
                  <tr>
                    <td colSpan="6" className="text-center py-8">
                      <LoadingSpinner text="Loading vehicles..." />
                    </td>
                  </tr>
                ) : filteredVehicles.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="text-center py-8 text-gray-400">
                      <Car className="h-8 w-8 mx-auto mb-2" />
                      <p className="text-sm">No stolen vehicles found</p>
                    </td>
                  </tr>
                ) : (
                  filteredVehicles.map((vehicle) => (
                    <tr key={vehicle.id} className={vehicle.is_active ? '' : 'bg-gray-50'}>
                      <td>
                        <span className="font-mono font-semibold text-gray-900">
                          {vehicle.plate_number}
                        </span>
                      </td>
                      <td>
                        <span className="capitalize">{vehicle.vehicle_type}</span>
                      </td>
                      <td>
                        {vehicle.is_active ? (
                          <span className="badge badge-danger">Active</span>
                        ) : (
                          <span className="badge badge-success">Resolved</span>
                        )}
                      </td>
                      <td>
                        <div className="flex items-center space-x-1 text-gray-600">
                          <Calendar className="h-3.5 w-3.5" />
                          <span>{new Date(vehicle.date_reported).toLocaleDateString()}</span>
                        </div>
                      </td>
                      <td>
                        <span className="text-gray-600 truncate max-w-xs block">
                          {vehicle.description || '-'}
                        </span>
                      </td>
                      <td className="text-right">
                        <div className="flex items-center justify-end space-x-1">
                          {vehicle.is_active && (
                            <button
                              onClick={() => handleResolve(vehicle.id)}
                              className="p-1.5 text-success-600 hover:bg-success-50 rounded"
                              title="Mark as Resolved"
                            >
                              <CheckCircle className="h-4 w-4" />
                            </button>
                          )}
                          <button
                            onClick={() => handleEdit(vehicle)}
                            className="p-1.5 text-primary-600 hover:bg-primary-50 rounded"
                            title="Edit"
                          >
                            <Edit className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(vehicle.id)}
                            className="p-1.5 text-danger-600 hover:bg-danger-50 rounded"
                            title="Delete"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Summary */}
        <div className="mt-3 flex justify-between text-xs text-gray-500">
          <span>
            Total: {vehicles.length} vehicles ({vehicles.filter(v => v.is_active).length} active)
          </span>
          <span>
            Showing {filteredVehicles.length} results
          </span>
        </div>
    </div>
  );
};

export default StolenVehiclesPage;
