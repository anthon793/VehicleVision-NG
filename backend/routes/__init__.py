"""
Routes package initialization.
Exports all route modules.
"""

from routes.auth_routes import router as auth_router
from routes.stolen_vehicles import router as stolen_vehicles_router
from routes.detection import router as detection_router
from routes.logs import router as logs_router

__all__ = [
    "auth_router",
    "stolen_vehicles_router", 
    "detection_router",
    "logs_router"
]
