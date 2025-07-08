# 🚀 Smart Video Analytics Platform

A Flask web application for **real-time video analytics** powered by **YOLOv8 object detection** and **Google Cloud Storage** integration.

It allows you to:

✅ Detect vehicles, people, and mobile phone usage in live RTSP streams  
✅ Define multiple Regions of Interest (ROIs) dynamically via a web interface  
✅ Log alerts (e.g., idle vehicles, unattended vehicles, mobile phone usage)  
✅ Save annotated event snapshots to Google Cloud Storage  
✅ View detection results and events live in the browser  

---

## 📂 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Setup](#-setup)
- [▶️ Running the Application](#️-running-the-application)
- [🌐 Endpoints](#-endpoints)
- [🛠️ Customization](#️-customization)
- [📸 Screenshots](#-screenshots)
- [📄 License](#-license)
- [🧠 Credits](#-credits)

---

## ✨ Features

### 🎯 Real-Time Object Detection
Uses Ultralytics YOLOv8 to detect:
- Vehicles (cars, buses, trucks, motorcycles)
- People
- Mobile phones

### 🗺️ Configurable ROIs
- Draw ROIs in the browser
- Save them with custom labels
- Filter detection results by ROI

### 🚨 Event Logging & Alerts
- Idle vehicle alerts (with adjustable thresholds)
- Unattended vehicle alerts
- Person using mobile phone alerts
- Logs with timestamps, saved frames, and alert levels

### ☁️ Cloud Storage Integration
- Automatically uploads event snapshots to Google Cloud Storage

### ⚡ Dynamic Camera Configuration
- Update Camera ID, Station Number, and Customer ID via the web UI

---

## 🏗️ Architecture

**Components:**

**Flask Backend**
- Streams RTSP video
- Runs YOLOv8 detection and tracking
- Processes event logic
- Exposes REST APIs for configuration, events, and ROI management

**Frontend**
- HTML5 / CSS3 / JavaScript single-page app
- Live video stream with overlaid detections
- ROI drawing canvas
- Event logs and inference logs

**Google Cloud Storage**
- Stores annotated frames of important events

---

## ⚙️ Setup

### 1️⃣ Clone the Repository & Install Dependencies

It is highly recommended to use a **virtual environment**:

```bash
git clone <your-repo-url>
cd <your-project-directory>

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Dependencies include:

Flask

OpenCV (opencv-python)

Ultralytics YOLOv8

google-cloud-storage

google-auth

Note: You must also install any additional dependencies YOLOv8 requires (e.g., PyTorch).

2️⃣ Configure Google Cloud Service Account & RTSP URL
Create a Service Account JSON Key with Storage Object Admin permission.

Replace SERVICE_ACCOUNT_INFO in app.py with your credentials:

python
Copy
Edit
SERVICE_ACCOUNT_INFO = {
    "type": "service_account",
    ...
}
In app.py, update your camera stream URL:

python
Copy
Edit
rtsp_url = "rtsp://<username>:<password>@<camera-address>/..."
▶️ Running the Application
Start the server:

bash
Copy
Edit
python app.py
The server will run at:

cpp
Copy
Edit
http://0.0.0.0:5000/
Open this URL in your browser to access the live stream and controls.

🌐 Endpoints
Endpoint	Purpose
/	Main web interface (stream, logs, ROI controls)
/stream	MJPEG video stream
/update_config	Update camera configuration
/update_rois	Save/update ROIs
/events_json	Retrieve event logs as JSON
/inference_json	Retrieve inference logs as JSON
/frame_dimensions	Get frame dimensions (for scaling ROIs)

🛠️ Customization
⚡ Adjust Detection Classes and Thresholds
In app.py, modify:

python
Copy
Edit
CONFIDENCE_THRESHOLD = 0.7

person_class_id = 0
cell_phone_class_id = 67
vehicle_class_ids = [1, 2, 3, 5, 7]
Change these IDs to detect other objects as needed.

🛑 Alert Timing and Distance
Adjust alert thresholds:

python
Copy
Edit
DWELL_TIME = 60          # seconds
WARNING_TIME = 45        # seconds
MOVE_THRESHOLD = 40      # pixels
📸 Screenshots
Add screenshots of your live stream page, event logs, and ROI drawing interface here.

📄 License
MIT License
(Or your preferred license)

🧠 Credits
Built with:

Ultralytics YOLOv8

Flask

Google Cloud Storage
