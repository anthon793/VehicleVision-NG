"""
Stolen Vehicle model for tracking reported stolen vehicles.
Defines the StolenVehicles table for admin management.
"""

from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from database.connection import Base
import enum


class VehicleType(str, enum.Enum):
    """Enumeration for supported vehicle types."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class StolenVehicle(Base):
    """
    Stolen Vehicle model for storing reported stolen vehicle information.
    
    Attributes:
        id: Primary key, auto-incremented
        plate_number: Normalized license plate number
        vehicle_type: Type of vehicle (car, truck, bus, motorcycle)
        date_reported: Date when the vehicle was reported stolen
        description: Optional description or notes
        is_active: Flag to indicate if the vehicle is still reported as stolen
    """
    __tablename__ = "stolen_vehicles"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    plate_number = Column(String(20), unique=True, index=True, nullable=False)
    vehicle_type = Column(Enum(VehicleType), nullable=False)
    date_reported = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(String(500), nullable=True)
    vehicle_color = Column(String(30), nullable=True)  # Expected color (e.g., "white", "black")
    is_active = Column(Integer, default=1)  # 1 = active, 0 = resolved
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<StolenVehicle(id={self.id}, plate='{self.plate_number}', type='{self.vehicle_type}')>"
