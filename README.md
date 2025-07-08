Smart Video Analytics Platform
This project is a Flask web application for real-time video analytics powered by YOLOv8 object detection and Google Cloud Storage integration.

It allows you to:

✅ Detect vehicles, people, and mobile phone usage in live RTSP streams
✅ Define multiple Regions of Interest (ROIs) dynamically via a web interface
✅ Log alerts (e.g., idle vehicles, unattended vehicles, mobile phone usage)
✅ Save annotated event snapshots to Google Cloud Storage
✅ View detection results and events live in the browser

📂 Table of Contents
Features

Architecture

Setup

Running the Application

Endpoints

Customization

Screenshots

License

✨ Features
Real-Time Object Detection
Uses Ultralytics YOLOv8 to detect:

Vehicles (cars, buses, trucks, motorcycles)

People

Mobile phones

Configurable ROIs

Draw ROIs in the browser.

Save them with custom labels.

Detection results are filtered by ROI.

Event Logging & Alerts

Idle vehicle alerts (with adjustable thresholds).

Unattended vehicle alerts.

Person using mobile phone alerts.

Logs with timestamps, saved frames, and alert levels.

Cloud Storage Integration

Snapshots of events are uploaded to Google Cloud Storage.

Dynamic Camera Configuration

Update Camera ID, Station Number, and Customer ID via UI.

🏗️ Architecture
Components:

Flask Backend:

Streams RTSP video.

Runs YOLOv8 detection & tracking.

Processes event logic.

Provides REST API for config, events, and ROI management.

Frontend:

HTML5/CSS3/JavaScript (single-page).

Video stream with overlaid detections.

ROI drawing canvas.

Event logs and inference logs.

Google Cloud Storage:

Stores annotated frames of important events.

⚙️ Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone <your-repo-url>
cd <your-project-directory>
2️⃣ Install Dependencies
It is highly recommended to use a virtual environment.

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
Dependencies:

Flask

OpenCV (opencv-python)

Ultralytics (YOLOv8)

google-cloud-storage

google-auth

Note: You must also install any additional dependencies YOLOv8 requires (PyTorch, etc).

3️⃣ Configure Google Cloud Service Account
Create a Service Account JSON Key with Storage Object Admin permission.

Replace SERVICE_ACCOUNT_INFO in app.py with your credentials.

4️⃣ Update RTSP URL
In app.py, configure:

python
Copy
Edit
rtsp_url = 'rtsp://<username>:<password>@<camera-address>/...'
▶️ Running the Application
bash
Copy
Edit
python app.py
The server will start on:

cpp
Copy
Edit
http://0.0.0.0:5000/
Live stream and controls will be available in the browser.

🌐 Endpoints
Endpoint	Purpose
/	Main web interface (stream, logs, ROI controls)
/stream	MJPEG video stream
/update_config	Update camera configuration (camera ID, station, etc.)
/update_rois	Save/update ROIs
/events_json	Retrieve event log in JSON
/inference_json	Retrieve inference log in JSON
/frame_dimensions	Get frame dimensions (for scaling ROI coordinates)

🛠️ Customization
⚡ Adjust Detection Classes and Thresholds
In app.py:

python
Copy
Edit
CONFIDENCE_THRESHOLD = 0.7
person_class_id = 0
cell_phone_class_id = 67
vehicle_class_ids = [1, 2, 3, 5, 7]
Modify these to detect additional objects.

🛑 Alert Timing and Distance
Change DWELL_TIME, WARNING_TIME, MOVE_THRESHOLD:

python
Copy
Edit
DWELL_TIME = 60  # seconds
WARNING_TIME = 45  # seconds
MOVE_THRESHOLD = 40  # pixels
📸 Screenshots
(Add screenshots of your live stream page, event logs, and ROI drawing interface here)

📄 License
MIT License.
(Or your preferred license)

🧠 Credits
Built with:

Ultralytics YOLOv8

Flask

Google Cloud Storage
