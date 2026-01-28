/**
 * Sidebar Navigation Component
 * Fixed sidebar with collapsible mobile menu
 */

import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { 
  Menu, X, LayoutDashboard, Camera, Upload, Database, 
  History, Users, LogOut, Shield, Image,
  ChevronLeft, ChevronRight
} from 'lucide-react';

const Sidebar = () => {
  const { user, logout, isAdmin } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navLinks = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/camera', label: 'Live Camera', icon: Camera },
    { to: '/upload', label: 'Upload Video', icon: Upload },
    { to: '/test', label: 'Test Image', icon: Image },
    { to: '/logs', label: 'Detection Logs', icon: History },
  ];

  // Admin-only links
  if (isAdmin()) {
    navLinks.push(
      { to: '/stolen-vehicles', label: 'Stolen Vehicles', icon: Database },
      { to: '/users', label: 'Manage Users', icon: Users }
    );
  }

  const isActive = (path) => location.pathname === path;

  const NavItem = ({ link, mobile = false }) => {
    const Icon = link.icon;
    const active = isActive(link.to);
    
    return (
      <Link
        to={link.to}
        onClick={() => mobile && setMobileOpen(false)}
        className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
          active
            ? 'bg-primary-600 text-white'
            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
        } ${collapsed && !mobile ? 'justify-center px-2' : ''}`}
        title={collapsed ? link.label : ''}
      >
        <Icon className={`h-[18px] w-[18px] flex-shrink-0 ${active ? 'text-white' : 'text-gray-500'}`} />
        {(!collapsed || mobile) && <span>{link.label}</span>}
      </Link>
    );
  };

  const getRoleLabel = (role) => {
    if (role === 'admin') return 'Administrator';
    if (role === 'user') return 'Operator';
    return role;
  };

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-md text-gray-600 hover:bg-gray-50 border border-gray-200"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Mobile Overlay */}
      {mobileOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile Sidebar */}
      <div className={`lg:hidden fixed inset-y-0 left-0 z-50 w-60 bg-white shadow-xl transform transition-transform duration-300 ${
        mobileOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <Shield className="h-4 w-4 text-white" />
              </div>
              <span className="font-bold text-lg text-gray-900">SVDS</span>
            </div>
            <button
              onClick={() => setMobileOpen(false)}
              className="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-2 space-y-0.5 overflow-y-auto">
            {navLinks.map((link) => (
              <NavItem key={link.to} link={link} mobile />
            ))}
          </nav>

          {/* User Section */}
          <div className="px-3 py-3 border-t border-gray-100">
            <div className="flex items-center gap-2.5 px-2 py-2">
              <div className="w-9 h-9 rounded-full bg-primary-600 flex items-center justify-center flex-shrink-0">
                <span className="text-white font-semibold text-sm">
                  {user?.username?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">{user?.username}</p>
                <p className="text-xs text-gray-500">{getRoleLabel(user?.role)}</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="w-full mt-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <LogOut className="h-4 w-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>

      {/* Desktop Sidebar */}
      <div className={`hidden lg:flex flex-col fixed inset-y-0 left-0 bg-white border-r border-gray-200 transition-all duration-300 z-30 ${
        collapsed ? 'w-16' : 'w-56'
      }`}>
        {/* Header */}
        <div className={`flex items-center h-14 px-3 border-b border-gray-100 ${collapsed ? 'justify-center' : 'gap-2'}`}>
          <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
            <Shield className="h-4 w-4 text-white" />
          </div>
          {!collapsed && (
            <span className="font-bold text-lg text-gray-900">SVDS</span>
          )}
        </div>

        {/* Collapse Button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-16 w-6 h-6 bg-white border border-gray-200 rounded-full shadow-sm flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-50 transition-colors"
        >
          {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
        </button>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-2 space-y-0.5 overflow-y-auto">
          {navLinks.map((link) => (
            <NavItem key={link.to} link={link} />
          ))}
        </nav>

        {/* User Section */}
        <div className={`px-2 py-3 border-t border-gray-100 ${collapsed ? 'px-2' : ''}`}>
          {collapsed ? (
            <div className="flex flex-col items-center gap-2">
              <div className="w-9 h-9 rounded-full bg-primary-600 flex items-center justify-center">
                <span className="text-white font-semibold text-sm">
                  {user?.username?.charAt(0).toUpperCase()}
                </span>
              </div>
              <button
                onClick={handleLogout}
                className="w-9 h-9 flex items-center justify-center text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                title="Logout"
              >
                <LogOut className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-2.5 px-2 py-2">
                <div className="w-9 h-9 rounded-full bg-primary-600 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-semibold text-sm">
                    {user?.username?.charAt(0).toUpperCase()}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">{user?.username}</p>
                  <p className="text-xs text-gray-500">{getRoleLabel(user?.role)}</p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="w-full mt-1 flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default Sidebar;
