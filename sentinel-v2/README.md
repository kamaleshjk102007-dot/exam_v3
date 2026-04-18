# SentinelEye � AI-Powered Exam Monitoring System

![Version](https://img.shields.io/badge/version-2.4.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-orange)

An intelligent, real-time exam cheating detection system powered by YOLOv8 computer vision. Detects mobile phones, suspicious behaviors, and exam violations through video analysis and live camera monitoring.

---

## ?? Features

### Core Capabilities
- **Real-Time Detection**: Live monitoring through RTSP/IP cameras with WebSocket streaming
- **Video Analysis**: Upload and process recorded exam footage with frame-by-frame analysis
- **Multi-Camera Support**: Monitor multiple examination rooms simultaneously
- **5 Detection Classes**:
  - ?? Phone Detection
  - ?? Hand Under Table
  - ?? Look Around (suspicious head movements)
  - ?? Wave (signaling gestures)
  - ?? Bend Over Desk (unusual postures)

### Technical Features
- **YOLOv8 AI Model**: State-of-the-art object detection
- **WebSocket Streaming**: Real-time frame delivery at up to 30 FPS
- **JWT Authentication**: Secure user sessions
- **MongoDB Integration**: Persistent storage for alerts and sessions
- **Progress Tracking**: Visual progress bars for video analysis
- **Violation Snapshots**: Automatic capture of detected violations
- **Alert Management**: Cooldown system to prevent spam
- **Responsive UI**: Works on desktop and mobile devices

### UI Modes
- **Dashboard**: Manage classrooms and view statistics
- **Video Mode**: Upload and analyze recorded footage
- **Real-Time Mode**: Live camera monitoring with camera switching
- **Demo Mode**: Fully functional simulation when backend is offline

---

## ??? Architecture

```
+-----------------+      +------------------+      +-----------------+
�   Frontend      �?----?�   FastAPI        �?----?�   MongoDB       �
�   (HTML/CSS/JS) �  WS  �   Backend        �      �   Database      �
+-----------------+      +------------------+      +-----------------+
                                  �
                                  ?
                         +------------------+
                         �   YOLOv8         �
                         �   Detection      �
                         +------------------+
                                  �
                                  ?
                         +------------------+
                         �   OpenCV         �
                         �   Video/Stream   �
                         +------------------+
```

---

## ?? Installation

### Prerequisites
- Python 3.8 or higher
- MongoDB 4.4+
- CUDA-capable GPU (recommended for real-time detection)
- Node.js (optional, only if you want frontend tooling outside the static app)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentineleye.git
cd sentineleye
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
copy .env.example .env
```

Then edit `.env` with your local settings.

**.env file structure:**
```env
APP_NAME=SentinelEye
DEBUG=false

# JWT
SECRET_KEY=change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=sentinel_eye

# CORS
ALLOWED_ORIGINS=["http://localhost:8000","http://127.0.0.1:8000","http://localhost:3000","http://localhost:5173"]

# Uploads
UPLOAD_DIR=uploads
MAX_FILE_SIZE_MB=500
ALLOWED_VIDEO_EXTENSIONS=[".mp4",".avi",".mov",".mkv"]

# Model
YOLO_MODEL=best.pt
YOLO_CONF=0.40
FRAME_SKIP=2
```

5. **Place the YOLOv8 model**
```bash
# Put best.pt inside backend/
```

6. **Start MongoDB**
```bash
mongod --dbpath /path/to/data
```

7. **Run the backend**
```bash
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

If Ultralytics cannot write to your default config directory on Windows, run with:

```powershell
$env:YOLO_CONFIG_DIR="D:\exam_v3\sentinel-v2\backend"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup

The frontend is static HTML served by FastAPI.

1. **Option 1: Serve via FastAPI**
   - Landing page: `http://127.0.0.1:8000/`
   - App page: `http://127.0.0.1:8000/app`

2. **Option 2: Use a local server**
```bash
cd backend/static
python -m http.server 8080
# Open http://localhost:8080
```

3. **Option 3: Open directly**
   - Open `backend/static/index.html` or `backend/static/landing.html` in a modern browser
   - Backend-powered features will not work without FastAPI

---

## ?? Usage

### 1. Login / Register
- **Default Demo Credentials**: Any email/password works in demo mode
- **Production**: Create an account through the registration tab

### 2. Dashboard
- View all classrooms and statistics
- Add new examination rooms
- Monitor total cameras and alerts

### 3. Video Analysis Mode
**Upload recorded exam footage:**
1. Select a classroom
2. Choose "Video Mode"
3. Drag & drop or select MP4/AVI/MOV file
4. Click "Analyze Video"
5. Watch real-time progress and detections
6. Review violation alerts and snapshots

### 4. Real-Time Monitoring
**Monitor live RTSP cameras:**
1. Select a classroom
2. Choose "Real-Time Mode"
3. Configure camera settings (IP, RTSP URL)
4. Click "Start Monitoring"
5. Switch between cameras
6. View live alerts and statistics
#### How to check whether realtime is working
1. Start the app with start.bat
2. Open http://127.0.0.1:<port>/health
3. Confirm the response contains:
   - "status": "ok"
   - "database_ready": true for full backend connectivity
4. In the dashboard, make sure the top badge shows BACKEND ONLINE
   - If it shows DEMO MODE, realtime is not connected
5. Open browser DevTools:
   - Console: check for fetch or WebSocket errors
   - Network -> WS: verify a realtime websocket opens and stays connected
6. Watch the backend terminal for new requests or stream activity when you start monitoring
7. Trigger a visible movement in front of the camera and check whether:
   - live frames update
   - alerts change
   - snapshots or counters update
#### Testing with VLC-compatible streams
- Use the original camera stream URL, not a lc://... link
- Supported test inputs are typically:
  - tsp://username:password@ip:554/stream
  - http://ip:port/video
  - https://.../stream.m3u8
- First test the URL in VLC
- If VLC cannot open the stream, SentinelEye usually cannot open it either
- Use only cameras and streams you own or are authorized to access
- Do not use random public or unauthorized CCTV links

### 5. Camera Management
- Add multiple cameras per room
- Configure IP addresses and RTSP streams
- Edit camera settings
- View online/offline status

---

## ?? API Endpoints

### Authentication
```http
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
```

### Classrooms
```http
GET    /api/classrooms/
POST   /api/classrooms/
GET    /api/classrooms/{id}
DELETE /api/classrooms/{id}
```

### Cameras
```http
GET    /api/cameras/classroom/{classroom_id}
POST   /api/cameras/
PATCH  /api/cameras/{id}
DELETE /api/cameras/{id}
```

### Detection
```http
POST /api/detection/upload
WS   /api/detection/ws/video/{video_id}
WS   /api/detection/ws/{session_id}
GET  /api/detection/snapshots
GET  /api/detection/snapshot/{filename}
```

### Alerts
```http
GET    /api/alerts/classroom/{classroom_id}
GET    /api/alerts/session/{session_id}
DELETE /api/alerts/classroom/{classroom_id}
```

---

## ?? Actual Project Structure

```
sentinel-v2/
+-- backend/
�   +-- database/
�   �   +-- connection.py
�   +-- routes/
�   �   +-- alerts.py
�   �   +-- auth.py
�   �   +-- cameras.py
�   �   +-- classrooms.py
�   �   +-- detection.py
�   +-- static/
�   �   +-- index.html
�   �   +-- landing.html
�   +-- .env.example
�   +-- best.pt
�   +-- config.py
�   +-- detector.py
�   +-- exam_detector_v3.py
�   +-- main.py
�   +-- requirements.txt
+-- evaluation/
+-- frontend/
�   +-- index.html
+-- README.md
+-- best.pt
```

---

## ?? Security Considerations

### Production Deployment
- [ ] Change default SECRET_KEY in .env
- [ ] Enable HTTPS/WSS (use nginx reverse proxy)
- [ ] Implement rate limiting on API endpoints
- [ ] Set up MongoDB authentication
- [ ] Use environment variables for sensitive data
- [ ] Configure CORS properly for your domain
- [ ] Regular security updates for dependencies
- [ ] Implement request size limits
- [ ] Add IP whitelisting for camera streams

### Data Privacy
- Video uploads are temporary (auto-delete recommended)
- Snapshots contain violation frames only
- Implement data retention policies
- Encrypt sensitive camera credentials
- Comply with local privacy laws (GDPR, etc.)

---

**Built with ?? for academic integrity**
