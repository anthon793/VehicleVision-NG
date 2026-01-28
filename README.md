## 🚢 Deploying to GitHub

### 1. Prepare for Deployment

- Ensure all sensitive data is removed from `.env` and only `.env.example` is committed.
- Add a `.gitignore` file to exclude environment files, model weights, uploads, and node_modules.
- Review code for hardcoded secrets or credentials.

### 2. Initialize Git and Push

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### 3. After Pushing

- Update the repository description and topics on GitHub.
- Add a license file if needed (see [Choose a License](https://choosealicense.com/)).
- Set up repository secrets for CI/CD if deploying to cloud platforms.

---
# PlateGuard: Stolen Vehicle & License Plate Detection

## About
PlateGuard is an open-source system for real-time vehicle and license plate detection, stolen vehicle alerts, and automated video analysis. Built with FastAPI, React, and YOLOv8, it empowers security, research, and smart city applications with modern AI-driven surveillance tools.

## 🎯 Overview

A comprehensive web-based system for detecting vehicles and license plates from live camera feeds and uploaded videos, with stolen vehicle identification capabilities. Built as a BSc research project MVP.

### Key Features
- **Live Camera Detection**: Real-time vehicle and license plate detection via webcam
- **Video Analysis**: Upload and process video files for batch detection
- **Stolen Vehicle Database**: Admin-managed database of reported stolen vehicles
- **Instant Alerts**: Visual alerts when a stolen vehicle is detected
- **Detection History**: Complete logging of all detections for analysis

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React SPA     │────▶│   FastAPI       │────▶│    MySQL        │
│   (Frontend)    │◀────│   (Backend)     │◀────│   Database      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   YOLOv8 +      │
                        │   EasyOCR       │
                        │   (AI Models)   │
                        └─────────────────┘
```

## 📋 Prerequisites

### Required Software
- **Python 3.10+** - Backend runtime
- **Node.js 18+** - Frontend build tools
- **MySQL 8.0+** - Database server
- **Git** - Version control

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (GPU optional but recommended)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd "c:\Users\ameha\Downloads\CV"

# Create Python virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install Python dependencies
cd backend
pip install -r requirements.txt
```

### 2. Database Setup

```sql
-- Open MySQL Command Line or Workbench
-- Run the schema file:
SOURCE c:/Users/ameha/Downloads/CV/backend/database/schema.sql;

-- Or manually create:
CREATE DATABASE stolen_vehicle_db;
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings:
# DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/stolen_vehicle_db
# SECRET_KEY=your-secure-random-key-here
```

### 4. Download AI Models

```bash
# The YOLOv8n model will auto-download on first run
# For custom plate detection model, place in:
# c:\Users\ameha\Downloads\CV\models\yolov8_plate.pt
```

### 5. Start Backend Server

```bash
cd backend
python main.py

# Or with uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Setup Frontend

```bash
# Open new terminal
cd frontend
npm install
npm run dev
```

### 7. Access the Application

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Default Login Credentials
- **Username**: `admin`
- **Password**: `admin123`

> ⚠️ **Change these credentials immediately in production!**

## 📁 Project Structure

```
project-root/
│
├── backend/
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Configuration settings
│   ├── requirements.txt     # Python dependencies
│   │
│   ├── auth/                # Authentication module
│   │   ├── utils.py         # JWT token utilities
│   │   ├── dependencies.py  # Auth dependencies
│   │   └── schemas.py       # Auth Pydantic schemas
│   │
│   ├── routes/              # API endpoints
│   │   ├── auth_routes.py   # Login/register
│   │   ├── stolen_vehicles.py # Vehicle CRUD
│   │   ├── detection.py     # Detection endpoints
│   │   └── logs.py          # History/stats
│   │
│   ├── services/            # Business logic
│   │   ├── detection.py     # YOLOv8 detection
│   │   ├── ocr.py           # License plate OCR
│   │   ├── video_processor.py # Video analysis
│   │   └── database_service.py # DB operations
│   │
│   ├── models/              # SQLAlchemy models
│   │   ├── user.py
│   │   ├── stolen_vehicle.py
│   │   └── detection_log.py
│   │
│   └── database/            # Database utilities
│       ├── connection.py    # Session management
│       └── schema.sql       # MySQL schema
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app component
│   │   ├── main.jsx         # React entry point
│   │   │
│   │   ├── pages/           # Page components
│   │   │   ├── LoginPage.jsx
│   │   │   ├── DashboardPage.jsx
│   │   │   ├── CameraPage.jsx
│   │   │   ├── UploadPage.jsx
│   │   │   ├── LogsPage.jsx
│   │   │   └── StolenVehiclesPage.jsx
│   │   │
│   │   ├── components/      # Reusable components
│   │   │   ├── Navbar.jsx
│   │   │   ├── DetectionOverlay.jsx
│   │   │   ├── StolenAlert.jsx
│   │   │   └── ...
│   │   │
│   │   ├── hooks/           # Custom React hooks
│   │   │   ├── useCamera.js
│   │   │   └── useDetection.js
│   │   │
│   │   ├── context/         # React context
│   │   │   └── AuthContext.jsx
│   │   │
│   │   └── services/        # API services
│   │       └── api.js
│   │
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── models/                  # AI model files
│   ├── yolov8n.pt          # Vehicle detection (auto-download)
│   └── yolov8_plate.pt     # Plate detection (custom)
│
└── README.md
```

## 🔐 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | User login |
| POST | `/auth/register` | Register user (admin) |
| GET | `/auth/me` | Get current user |
| POST | `/auth/init-admin` | Initialize admin account |

### Stolen Vehicles (Admin)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stolen-vehicles/` | List all |
| POST | `/stolen-vehicles/` | Add new |
| PUT | `/stolen-vehicles/{id}` | Update |
| DELETE | `/stolen-vehicles/{id}` | Delete |
| POST | `/stolen-vehicles/{id}/resolve` | Mark resolved |
| GET | `/stolen-vehicles/check/{plate}` | Check plate |

### Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detection/frame` | Detect in single frame |
| POST | `/detection/upload-video` | Upload video file |
| POST | `/detection/process-video/{filename}` | Process video |
| DELETE | `/detection/video/{filename}` | Delete video |

### Logs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/logs/` | Get all logs |
| GET | `/logs/statistics` | Get stats |
| GET | `/logs/recent-stolen` | Recent alerts |
| GET | `/logs/today` | Today's detections |

## 🧠 Computer Vision Pipeline

```
Input Frame
     │
     ▼
┌─────────────────┐
│ Vehicle Detection│ ◀─── YOLOv8n (COCO classes: car, truck, bus, motorcycle)
│ (Stage 1)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Plate Detection │ ◀─── YOLOv8 (custom) or Region Estimation
│ (Stage 2)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Preprocess│ ◀─── Grayscale, CLAHE, Adaptive Threshold
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OCR Extraction  │ ◀─── EasyOCR (English)
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Plate Normalize │ ◀─── Uppercase, remove spaces/hyphens
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Database Match  │ ◀─── Compare against stolen plates
│                 │
└────────┬────────┘
         │
         ▼
    Detection Result
```

## 🇳🇬 Nigerian License Plate Support

The OCR system is optimized for Nigerian license plate formats:
- **Standard**: ABC 123 XY
- **Lagos State**: LAG 234 ABC
- **Federal Capital**: ABJ 567 CD

### Plate Normalization Rules
1. Convert to uppercase
2. Remove spaces, hyphens, dots
3. Apply OCR corrections (O→0, I→1 in digit positions)
4. Validate 6-10 alphanumeric characters

## ⚙️ Configuration Options

### Backend (.env)
```ini
# Database
DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/stolen_vehicle_db

# JWT Security
SECRET_KEY=your-256-bit-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Detection
DETECTION_CONFIDENCE=0.5
FRAME_SKIP=5

# Paths
VEHICLE_MODEL_PATH=../models/yolov8n.pt
PLATE_MODEL_PATH=../models/yolov8_plate.pt
```

### Frontend (vite.config.js)
```javascript
server: {
  port: 3000,
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

## 📊 Evaluation Metrics

For academic evaluation (Chapter 5), the system logs:
- Detection confidence scores
- OCR confidence scores
- Processing time per frame
- Match accuracy (true positive/negative rates)
- System latency measurements

Access via:
```bash
GET /logs/statistics
GET /logs/by-date?start_date=2024-01-01&end_date=2024-12-31
```

## 🐛 Troubleshooting

### Common Issues

**1. Camera not working**
- Ensure browser has camera permissions
- Use HTTPS in production (required for camera API)
- Check if camera is used by another application

**2. Models not loading**
- First run downloads YOLOv8n automatically (~6MB)
- Check internet connection for initial download
- Verify model paths in .env

**3. MySQL connection error**
```bash
# Check MySQL service is running
net start mysql

# Verify credentials in .env
mysql -u root -p -e "SELECT 1"
```

**4. CORS errors**
- Ensure frontend URL is in `allow_origins` in main.py
- Check proxy configuration in vite.config.js

**5. Slow detection**
- Increase FRAME_SKIP value
- Reduce detection interval in frontend settings
- Consider GPU acceleration with CUDA

## 🔒 Security Considerations

- JWT tokens expire in 30 minutes (configurable)
- Passwords hashed with bcrypt
- Role-based access control (admin/user)
- CORS restricted to specified origins
- Input validation on all endpoints

## 📝 Academic Notes

This system is designed for BSc research purposes:
- **Modular architecture** allows individual component testing
- **Comprehensive logging** supports quantitative analysis
- **RESTful API** demonstrates modern web development practices
- **Two-stage detection** pipeline is academically defensible
- **Code comments** explain design decisions

## 📜 License

Academic/Research Use Only - Not for commercial deployment.

## 👥 Contributors

BSc Research Project Team

---

**Version**: 1.0.0  
**Last Updated**: January 2026
