-- MySQL Database Schema for Stolen Vehicle Detection System
-- Run this script to create the database and tables

-- Create database
CREATE DATABASE IF NOT EXISTS stolen_vehicle_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE stolen_vehicle_db;

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_role (role)
) ENGINE=InnoDB;

-- Stolen vehicles table for admin management
CREATE TABLE IF NOT EXISTS stolen_vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) NOT NULL UNIQUE,
    vehicle_type ENUM('car', 'truck', 'bus', 'motorcycle') NOT NULL,
    date_reported TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description VARCHAR(500) NULL,
    is_active TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_plate_number (plate_number),
    INDEX idx_vehicle_type (vehicle_type),
    INDEX idx_is_active (is_active),
    INDEX idx_date_reported (date_reported)
) ENGINE=InnoDB;

-- Detection logs table for storing all detections
CREATE TABLE IF NOT EXISTS detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detected_plate VARCHAR(20) NOT NULL,
    vehicle_type VARCHAR(20) NULL,
    match_status ENUM('stolen', 'not_stolen', 'unknown') DEFAULT 'unknown',
    confidence_score FLOAT NULL,
    ocr_confidence FLOAT NULL,
    source_type VARCHAR(20) NULL,
    source_filename VARCHAR(255) NULL,
    frame_number INT NULL,
    processing_time_ms FLOAT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Geolocation fields (new)
    latitude FLOAT NULL,
    longitude FLOAT NULL,
    location_accuracy FLOAT NULL,
    location_name VARCHAR(255) NULL,
    -- Processing metadata (new)
    track_id INT NULL,
    is_validated TINYINT(1) DEFAULT 0,
    processing_level VARCHAR(20) NULL,
    quality_score FLOAT NULL,
    INDEX idx_detected_plate (detected_plate),
    INDEX idx_match_status (match_status),
    INDEX idx_timestamp (timestamp),
    INDEX idx_source_type (source_type),
    INDEX idx_location (latitude, longitude)
) ENGINE=InnoDB;

-- Insert default admin user (password: admin123)
-- Password hash for 'admin123' using bcrypt
INSERT INTO users (username, password_hash, role) VALUES 
('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.HXzUzG0pUU0XWu', 'admin')
ON DUPLICATE KEY UPDATE username = username;

-- Insert sample stolen vehicles for testing
INSERT INTO stolen_vehicles (plate_number, vehicle_type, description) VALUES 
('LAG234ABC', 'car', 'Blue Toyota Camry, reported stolen 2024-01-10'),
('KJA567XY', 'motorcycle', 'Red Honda motorcycle'),
('ABJ123DE', 'truck', 'White Mercedes truck'),
('OYO890FG', 'bus', 'Yellow commercial bus')
ON DUPLICATE KEY UPDATE plate_number = plate_number;

-- Create view for quick stolen plate lookup
CREATE OR REPLACE VIEW active_stolen_plates AS
SELECT plate_number, vehicle_type, date_reported
FROM stolen_vehicles
WHERE is_active = 1;

-- Create view for detection statistics
CREATE OR REPLACE VIEW detection_stats AS
SELECT 
    DATE(timestamp) as detection_date,
    COUNT(*) as total_detections,
    SUM(CASE WHEN match_status = 'stolen' THEN 1 ELSE 0 END) as stolen_matches,
    SUM(CASE WHEN match_status = 'not_stolen' THEN 1 ELSE 0 END) as not_stolen,
    AVG(processing_time_ms) as avg_processing_time
FROM detection_logs
GROUP BY DATE(timestamp)
ORDER BY detection_date DESC;

-- Stored procedure to check plate status
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS check_plate_status(IN p_plate VARCHAR(20))
BEGIN
    SELECT 
        CASE WHEN COUNT(*) > 0 THEN TRUE ELSE FALSE END as is_stolen,
        sv.plate_number,
        sv.vehicle_type,
        sv.date_reported,
        sv.description
    FROM stolen_vehicles sv
    WHERE sv.plate_number = UPPER(REPLACE(REPLACE(p_plate, ' ', ''), '-', ''))
    AND sv.is_active = 1;
END //
DELIMITER ;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON stolen_vehicle_db.* TO 'your_user'@'localhost';
-- FLUSH PRIVILEGES;
