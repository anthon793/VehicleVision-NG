"""
Models package initialization.
Exports all database models.
"""

from models.user import User, UserRole
from models.stolen_vehicle import StolenVehicle, VehicleType
from models.detection_log import DetectionLog, MatchStatus

__all__ = [
    "User", "UserRole",
    "StolenVehicle", "VehicleType", 
    "DetectionLog", "MatchStatus"
]
