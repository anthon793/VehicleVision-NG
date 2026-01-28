"""
Stolen Vehicle Detection System - Main FastAPI Application
==========================================================

A web-based system for detecting vehicles and license plates from camera feeds
and uploaded videos, with stolen vehicle identification capabilities.

Developed as a BSc research project MVP.

Author: Research Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import uvicorn

from config import settings
from database import init_db, engine, Base
from routes import auth_router, stolen_vehicles_router, detection_router, logs_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events handler.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Stolen Vehicle Detection System...")
    
    # Initialize database tables
    logger.info("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    
    # Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    
    # Pre-load models (optional, improves first request latency)
    try:
        logger.info("Pre-loading detection models...")
        from services import get_detection_service
        get_detection_service()
        logger.info("Detection models loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-loading failed (will load on first request): {str(e)}")
    
    logger.info("System startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Stolen Vehicle Detection System...")


# API Tags for Swagger grouping
tags_metadata = [
    {
        "name": "Health",
        "description": "System health check endpoints for monitoring and status verification.",
    },
    {
        "name": "Authentication",
        "description": "User authentication endpoints - login, register, and token management.",
    },
    {
        "name": "Stolen Vehicles",
        "description": "Manage the stolen vehicle database. **Requires Admin privileges.**",
    },
    {
        "name": "Detection",
        "description": "Vehicle and license plate detection endpoints. Supports camera feeds, image uploads, and video processing.",
    },
    {
        "name": "Detection Logs",
        "description": "View detection history and statistics.",
    },
]

# Create FastAPI application with enhanced Swagger documentation
app = FastAPI(
    title="Stolen Vehicle Detection System API",
    description="""
## 🚗 Stolen Vehicle Detection System (SVDS)

A real-time vehicle and license plate detection system with stolen vehicle identification capabilities.

### 🔑 Authentication
All endpoints (except health checks) require JWT authentication. 
1. Register or login via `/auth/login`
2. Use the token in the **Authorize** button above
3. Format: `Bearer <your_token>`

### 📋 Features

| Feature | Description |
|---------|-------------|
| **Live Camera Detection** | Real-time detection from webcam feeds |
| **Image Upload** | Process uploaded images for vehicle detection |
| **Video Analysis** | Upload and analyze video files |
| **Stolen Vehicle Matching** | Automatic matching against stolen database |
| **Email Alerts** | Automatic email notifications on detection |
| **Detection Logs** | Complete history and analytics |

### 🛠️ Technical Stack

- **AI/ML**: YOLOv8 (vehicle detection), Roboflow (plate detection), Google Vision OCR
- **Backend**: FastAPI, Python 3.10+
- **Database**: SQLite/MySQL with SQLAlchemy ORM
- **Security**: JWT authentication, bcrypt password hashing

### 📊 Detection Workflow

```
1. Image/Video Input → 2. Vehicle Detection (YOLOv8)
                     → 3. Plate Detection (Roboflow)
                     → 4. OCR (Google Vision)
                     → 5. Database Matching
                     → 6. Alert if Stolen
```

### 👥 User Roles

| Role | Permissions |
|------|-------------|
| **Admin** | Full access - manage stolen vehicles, view all logs |
| **User** | Detection capabilities, view own detection history |

---
**Version**: 1.0.0 | **Contact**: support@svds.com
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "SVDS Support Team",
        "email": "support@svds.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003", "http://localhost:3004", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002", "http://127.0.0.1:3003", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions gracefully."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.SECRET_KEY == "your-super-secret-key-change-in-production" else "An error occurred"
        }
    )


# Include routers
app.include_router(auth_router)
app.include_router(stolen_vehicles_router)
app.include_router(detection_router)
app.include_router(logs_router)


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Stolen Vehicle Detection System",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns system status and component health.
    """
    services_status = {
        "database": "unknown",
        "detection_model": "unknown",
        "ocr_service": "unknown"
    }
    
    # Check database
    try:
        from database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        services_status["database"] = "healthy"
    except Exception as e:
        services_status["database"] = f"unhealthy: {str(e)}"
    
    # Check detection service
    try:
        from services import get_detection_service
        service = get_detection_service()
        services_status["detection_model"] = "healthy" if service.vehicle_model else "unavailable"
    except Exception as e:
        services_status["detection_model"] = f"unhealthy: {str(e)}"
    
    # Check OCR service
    try:
        from services import get_ocr_service
        service = get_ocr_service()
        services_status["ocr_service"] = "healthy" if service.reader else "unavailable"
    except Exception as e:
        services_status["ocr_service"] = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all(
        s == "healthy" for s in services_status.values()
    ) else "degraded"
    
    return {
        "status": overall_status,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services_status
    }


# Run with uvicorn when executed directly
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
