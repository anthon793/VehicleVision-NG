/**
 * Layout Component
 * Wraps pages with sidebar navigation
 */

import React from 'react';
import Sidebar from './Sidebar';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      
      {/* Main Content Area - offset for sidebar (w-56 = 224px) */}
      <div className="lg:pl-56 transition-all duration-300">
        <main className="min-h-screen">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
