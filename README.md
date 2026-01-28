git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/anthon793/VehicleVision-NG.git
git push -u origin main

# VehicleVision-NG

## Intelligent Vehicle & License Plate Detection for Nigeria

VehicleVision-NG is an open-source platform for real-time vehicle and license plate detection, stolen vehicle alerts, and video analytics, tailored for Nigerian roads and plate formats. Built with FastAPI, React, and YOLOv8, it empowers security, research, and smart city applications with modern AI-driven surveillance tools.

---

## 🚗 Features

- **Live Camera Detection:** Real-time vehicle and license plate detection via webcam or IP camera
- **Video Upload & Analysis:** Batch process video files for detection and logging
- **Stolen Vehicle Alerts:** Instantly flag and alert on matches from a managed stolen vehicle database
- **Nigerian Plate Support:** Optimized OCR and validation for Nigerian license plate formats
- **Admin Dashboard:** Manage users, view detection logs, and monitor system health
- **Email Notifications:** Automated alerts for stolen vehicle detections
- **RESTful API:** Modern, documented endpoints for integration

---

## 🏗️ Tech Stack

- **Backend:** FastAPI, SQLAlchemy, MySQL/SQLite
- **Frontend:** React, Vite, TailwindCSS
- **AI Models:** YOLOv8 (vehicle/plate detection), EasyOCR
- **Email:** Brevo (Sendinblue) or SMTP

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/VehicleVision-NG.git
cd VehicleVision-NG
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
cp .env.example .env  # Edit .env with your secrets
```

### 3. Database
- For MySQL: create a database and update `DATABASE_URL` in `.env`.
- For SQLite: default config works out of the box.

### 4. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```

### 5. Run the Backend
```bash
cd backend
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access the App
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## ⚙️ Configuration

All environment variables are documented in `backend/.env.example`.

---

## 📦 Project Structure

```
VehicleVision-NG/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── ...
├── frontend/
│   ├── src/
│   ├── ...
├── models/
├── README.md
```

---

## 📜 License

Academic/Research Use Only – Not for commercial deployment.

---

## 👤 Contributors

BSc Research Project Team

---

**Version:** 1.0.0  
**Last Updated:** January 2026
