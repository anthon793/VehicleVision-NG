/**
 * Dashboard Page Component
 * Main landing page with statistics and recent activity
 */

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { logsAPI, stolenVehiclesAPI, healthAPI } from '../services/api';
import StatsCard from '../components/StatsCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { 
  Camera, Upload, AlertTriangle, Car, CheckCircle, 
  Clock, TrendingUp, Database, Activity
} from 'lucide-react';

const DashboardPage = () => {
  const { user, isAdmin } = useAuth();
  const [stats, setStats] = useState(null);
  const [recentStolen, setRecentStolen] = useState([]);
  const [stolenCount, setStolenCount] = useState(0);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [statsData, recentData, vehiclesData, healthData] = await Promise.all([
          logsAPI.getStatistics(),
          logsAPI.getRecentStolen(5),
          stolenVehiclesAPI.getAll(),
          healthAPI.check()
        ]);

        setStats(statsData);
        setRecentStolen(recentData);
        setStolenCount(vehiclesData.length);
        setSystemHealth(healthData);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" text="Loading dashboard..." />
      </div>
    );
  }

  return (
    <div className="p-4 lg:p-6">
        {/* Welcome Section */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">
            Welcome back, {user?.username}!
          </h1>
          <p className="text-sm text-gray-600 mt-0.5">
            Here's what's happening with your vehicle detection system today.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
          <Link
            to="/camera"
            className="card hover:shadow-md transition-shadow flex items-center space-x-3 group"
          >
            <div className="p-3 bg-primary-100 rounded-lg group-hover:bg-primary-200 transition-colors">
              <Camera className="h-6 w-6 text-primary-600" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-900">Live Camera Detection</h3>
              <p className="text-xs text-gray-500">Start real-time plate scanning</p>
            </div>
          </Link>

          <Link
            to="/upload"
            className="card hover:shadow-md transition-shadow flex items-center space-x-3 group"
          >
            <div className="p-3 bg-success-100 rounded-lg group-hover:bg-success-200 transition-colors">
              <Upload className="h-6 w-6 text-success-600" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-900">Upload Video</h3>
              <p className="text-xs text-gray-500">Analyze recorded footage</p>
            </div>
          </Link>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          <StatsCard
            title="Total Detections"
            value={stats?.total_detections || 0}
            icon={TrendingUp}
            color="primary"
          />
          <StatsCard
            title="Stolen Matches"
            value={stats?.stolen_matches || 0}
            icon={AlertTriangle}
            color="danger"
          />
          <StatsCard
            title="Clear Vehicles"
            value={stats?.not_stolen || 0}
            icon={CheckCircle}
            color="success"
          />
          <StatsCard
            title="In Database"
            value={stolenCount}
            icon={Database}
            color="warning"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Recent Stolen Detections */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <h2 className="card-header mb-0">Recent Stolen Detections</h2>
              <Link to="/logs" className="text-xs text-primary-600 hover:text-primary-700">
                View all →
              </Link>
            </div>
            
            {recentStolen.length > 0 ? (
              <div className="space-y-2">
                {recentStolen.map((log) => (
                  <div
                    key={log.id}
                    className="flex items-center justify-between p-2.5 bg-danger-50 rounded-md border border-danger-100"
                  >
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="h-4 w-4 text-danger-500" />
                      <div>
                        <p className="font-mono text-sm font-bold text-danger-700">
                          {log.detected_plate}
                        </p>
                        <p className="text-xs text-danger-500">
                          {log.vehicle_type || 'Unknown'} • {log.source_type}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-gray-500">
                        {new Date(log.timestamp).toLocaleDateString()}
                      </p>
                      <p className="text-xs text-gray-400">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-gray-500">
                <CheckCircle className="h-10 w-10 mx-auto mb-2 text-success-300" />
                <p className="text-sm">No stolen vehicles detected recently</p>
              </div>
            )}
          </div>

          {/* System Status */}
          <div className="card">
            <h2 className="card-header">System Status</h2>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2.5 bg-gray-50 rounded-md">
                <div className="flex items-center space-x-2">
                  <Activity className={`h-4 w-4 ${systemHealth?.status === 'healthy' ? 'text-success-500' : 'text-yellow-500'}`} />
                  <span className="text-sm font-medium">Overall Status</span>
                </div>
                <span className={`badge ${systemHealth?.status === 'healthy' ? 'badge-success' : 'badge-warning'}`}>
                  {systemHealth?.status || 'Unknown'}
                </span>
              </div>

              {systemHealth?.services && Object.entries(systemHealth.services).map(([service, status]) => (
                <div key={service} className="flex items-center justify-between p-2.5 bg-gray-50 rounded-md">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${status === 'healthy' ? 'bg-success-500' : 'bg-yellow-500'}`} />
                    <span className="text-sm capitalize">{service.replace('_', ' ')}</span>
                  </div>
                  <span className={`text-xs ${status === 'healthy' ? 'text-success-600' : 'text-yellow-600'}`}>
                    {status}
                  </span>
                </div>
              ))}

              <div className="pt-2 border-t border-gray-200">
                <p className="text-xs text-gray-400">
                  Last updated: {systemHealth?.timestamp ? new Date(systemHealth.timestamp).toLocaleString() : 'N/A'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Admin Section */}
        {isAdmin() && (
          <div className="mt-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-3">Admin Actions</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              <Link
                to="/stolen-vehicles"
                className="card hover:shadow-md transition-shadow flex items-center space-x-3"
              >
                <Car className="h-6 w-6 text-danger-500" />
                <div>
                  <h3 className="text-sm font-medium text-gray-900">Manage Stolen Vehicles</h3>
                  <p className="text-xs text-gray-500">{stolenCount} vehicles registered</p>
                </div>
              </Link>

              <Link
                to="/logs"
                className="card hover:shadow-md transition-shadow flex items-center space-x-3"
              >
                <Clock className="h-6 w-6 text-primary-500" />
                <div>
                  <h3 className="text-sm font-medium text-gray-900">Detection History</h3>
                  <p className="text-xs text-gray-500">{stats?.total_detections || 0} total records</p>
                </div>
              </Link>
            </div>
          </div>
        )}
    </div>
  );
};

export default DashboardPage;
