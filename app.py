import os
import time
import cv2
import threading
from flask import Flask, Response, render_template_string, stream_with_context
from ultralytics import YOLO
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
from flask import request, jsonify
from collections import defaultdict

# Fix OpenMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Replace with your actual service account details
SERVICE_ACCOUNT_INFO = {"your_gcp_bucket_credentials"}

GCS_BUCKET_NAME = "your_bucket_name"
# GCS_FOLDER = "videos-dev"

credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)
gcs_client = storage.Client(credentials=credentials, project=SERVICE_ACCOUNT_INFO["project_id"])
bucket = gcs_client.bucket(GCS_BUCKET_NAME)

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

rtsp_url = 'rtsp_link'

CONFIDENCE_THRESHOLD = 0.7  

camera_config = {
    "camera_id": "CAM1",
    "station_number": "Station1"
}

person_class_id = 0
cell_phone_class_id = 67
vehicle_class_ids = [1, 2, 3, 5, 7]

class_names = model.names
tracked_vehicles = {}

DWELL_TIME = 60
WARNING_TIME = 45
MOVE_THRESHOLD = 40

event_log = []
MAX_EVENTS = 20

inference_log = []
MAX_INFERENCES = 20

# Define multiple ROIs for gas station
ROIs = {}

# Store the actual frame dimensions for coordinate scaling
frame_dimensions = {"width": 640, "height": 480}  # Default values, will be updated

def get_roi_label(x1, y1, x2, y2):
    for label, (top_left, bottom_right) in ROIs.items():
        rx1, ry1 = top_left
        rx2, ry2 = bottom_right

        # Return label if there is any intersection between the bounding box and the ROI
        if x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1:
            return label
    return "Unknown"

def save_event_frame(frame, event_type, track_id, roi_label="Unknown"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    camera_id = camera_config.get("camera_id", "CAM1")
    station_number = camera_config.get("station_number", "Station1")
    customer_id = camera_config.get("customer_id", "Customer1")
    filename = f"{event_type}_ID{track_id}_{roi_label}_{customer_id}_{camera_id}_{station_number}_{timestamp}.jpg"

    blob_path = f"{GCS_FOLDER}/{filename}"

    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        print("Failed to encode image.")
        return None

    blob = bucket.blob(blob_path)
    blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')
    print(f"[GCS] Uploaded: {blob_path}")
    return filename

def generate_frames():
    global frame_dimensions
    results = model.track(
        source=rtsp_url,
        show=False,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True
    )

    prev_time = time.time()

    for result in results:
        frame = result.orig_img
        current_time = time.time()

        # Update frame dimensions for coordinate scaling
        frame_dimensions["height"], frame_dimensions["width"] = frame.shape[:2]

        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        if result.boxes.id is None:
            continue

        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        roi_person_count = {label: 0 for label in ROIs}
        roi_vehicle_count = {label: 0 for label in ROIs}
        persons = []
        cell_phones = []

        for label, (top_left, bottom_right) in ROIs.items():
            # Ensure integer coordinates
            x1, y1 = map(int, top_left)
            x2, y2 = map(int, bottom_right)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"ROI: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        for track_id, cls, box, conf in zip(ids, classes, boxes, confidences):
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = class_names[cls]

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f'{label} ID: {track_id} ({conf:.2f})'
            cv2.putText(frame, label_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            roi_label = get_roi_label(x1, y1, x2, y2)
            if roi_label == "Unknown":
                continue

            if cls == person_class_id:
                roi_person_count[roi_label] += 1
                persons.append((track_id, (x1, y1, x2, y2)))
            elif cls == cell_phone_class_id:
                cell_phones.append((track_id, (x1, y1, x2, y2)))
            elif cls in vehicle_class_ids:
                roi_vehicle_count[roi_label] += 1
                
                #Alert for Vehicle-idle
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = {
                        'start_time': current_time,
                        'last_attended_time': current_time,
                        'bbox': (cx, cy),
                        'alert_level': 0
                    }
                else:
                    prev_cx, prev_cy = tracked_vehicles[track_id]['bbox']
                    distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                    if distance > MOVE_THRESHOLD:
                        tracked_vehicles[track_id]['start_time'] = current_time
                        tracked_vehicles[track_id]['last_attended_time'] = current_time
                        tracked_vehicles[track_id]['bbox'] = (cx, cy)

                dwell_duration = current_time - tracked_vehicles[track_id]['start_time']
                interval = int(dwell_duration // 180)

                unattended_duration = current_time - tracked_vehicles[track_id]['last_attended_time']

                # New: Check for unattended vehicle > 30 seconds
                if unattended_duration > 30:
                    attended = False
                    for pid, (px1, py1, px2, py2) in persons:
                        if get_roi_label(x1, y1, x2, y2) == get_roi_label(px1, py1, px2, py2):
                            attended = True
                            tracked_vehicles[track_id]['last_attended_time'] = current_time
                            # Reset alert level and remove previous unattended alerts if any
                            if 'unattended_alert_level' in tracked_vehicles[track_id]:
                                del tracked_vehicles[track_id]['unattended_alert_level']
                                # Remove all previous unattended alerts related to this vehicle
                                event_log[:] = [e for e in event_log if f"Vehicle {track_id} unattended" not in e]
                            break

                    if not attended:
                        unattended_interval = int(unattended_duration // 30)
                        last_alert = tracked_vehicles[track_id].get('unattended_alert_level', -1)

                        if unattended_interval > last_alert:
                            alert_color = (0, 165, 255)
                            unattended_msg = f'ALERT: Vehicle {track_id} unattended >{unattended_interval * 30}s'
                            cv2.putText(frame, unattended_msg, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2)
                            print(unattended_msg)
                            event_log.append(f"{roi_label}: {unattended_msg}")
                            filename = save_event_frame(frame, "unattended_vehicle", track_id, roi_label)
                            event_log.append(f"{roi_label}: {unattended_msg} (Frame: {filename})")


                if interval > tracked_vehicles[track_id]['alert_level']:
                    alert_color = (0, 0, 255)
                    alert_msg = f'ALERT: {label} {track_id} idle for {interval * 3} minutes'
                    cv2.putText(frame, alert_msg, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2)
                    print(alert_msg)
                    event_log.append(f"{roi_label}: {alert_msg} (Frame: {filename})")
                    filename = save_event_frame(frame, "idle_vehicle", track_id, roi_label)
                    tracked_vehicles[track_id]['alert_level'] = interval
                elif dwell_duration >= WARNING_TIME:
                    alert_color = (0, 255, 255)
                else:
                    alert_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 2)
                cv2.putText(frame, f'{label} ID: {track_id}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        
        #Alert for Person using Cell-Phone
        for pid, p_box in persons:
            px1, py1, px2, py2 = p_box
            for cid, c_box in cell_phones:
                cx1, cy1, cx2, cy2 = c_box

                if (px1 < cx2 and px2 > cx1 and py1 < cy2 and py2 > cy1):
                    color = (255, 0, 0)
                    alert_msg = f'ALERT: Person {pid} using mobile phone'

                    # Determine the ROI for the person
                    person_roi = get_roi_label(px1, py1, px2, py2)
                    if person_roi != "Unknown":
                        alert_msg += f' in ROI: {person_roi}'

                    cv2.putText(frame, alert_msg, (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(alert_msg)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                    event_log.append(f"{person_roi}: {alert_msg} (Frame: {filename})")
                    filename = save_event_frame(frame, "mobile_user", pid, person_roi)

        while len(event_log) > MAX_EVENTS:
            event_log.pop(0)

        text_color = (255, 255, 255)
        y_offset = 30
        for label in roi_person_count:
            cv2.putText(frame, f"{label} - People: {roi_person_count[label]}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 30
            cv2.putText(frame, f"{label} - Vehicles: {roi_vehicle_count[label]}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 40


        cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        frame_summary = [class_names[cls] for cls in classes]
        summary_text = f"{len(frame_summary)}: " + ', '.join(frame_summary)
        inference_time = result.speed['inference']
        fps = 1000 / inference_time if inference_time > 0 else 0
        fps_text = f"FPS: {fps:.1f}"

        cv2.putText(frame, fps_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        summary_text += f", {inference_time:.1f}ms, {fps_text}"
        print(summary_text)
        inference_log.append(summary_text)

        while len(inference_log) > MAX_INFERENCES:
            inference_log.pop(0)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SST Vision</title>
    <style>
        /* Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

        :root {
            --primary-color: #3f51b5;
            --secondary-color: #ff4081;
            --bg-light: #f9f9f9;
            --card-bg: #ffffff;
            --text-dark: #333333;
            --text-light: #555555;
            --border-radius: 12px;
            --shadow-soft: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--bg-light);
            color: var(--text-dark);
            line-height: 1.6;
        }

        h2, h4 {
            color: var(--primary-color);
            margin-bottom: 8px;
            font-weight: 500;
        }

        #video-container {
            max-width: 960px;
            margin: 30px auto 20px auto;
            padding: 0 20px;
            text-align: center;
        }

        #video-container h2 {
            font-size: 28px;
        }

        #video-container img {
            width: 100%;
            max-height: 500px;
            object-fit: cover;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-soft);
            transition: transform 0.3s ease;
        }

        #video-container img:hover {
            transform: scale(1.02);
        }

        .box-wrapper {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-bottom: 40px;
        }

        .box {
            background-color: var(--card-bg);
            flex: 1 1 320px;
            max-width: 400px;
            padding: 20px;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-soft);
            overflow-y: auto;
            max-height: 250px;
        }

        .box h4 {
            font-size: 20px;
            margin-bottom: 12px;
        }

        ul {
            list-style: none;
            padding-left: 0;
        }

        li {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: var(--border-radius);
            transition: background 0.2s ease;
            font-size: 15px;
            gap: 8px;
        }

        li:hover {
            background-color: #f1f1f1;
        }

        .green-dot::before,
        .red-dot::before,
        .yellow-dot::before {
            font-size: 18px;
        }

        .green-dot::before {
            content: "🟢";
        }

        .red-dot::before {
            content: "🔴";
        }

        .yellow-dot::before {
            content: "🟠";
        }

        /* Custom scrollbar */
        .box::-webkit-scrollbar {
            width: 6px;
        }
        .box::-webkit-scrollbar-track {
            background: transparent;
        }
        .box::-webkit-scrollbar-thumb {
            background-color: var(--primary-color);
            border-radius: 3px;
        }

        @media (max-width: 600px) {
            #video-container h2 { font-size: 24px; }
            .box { padding: 15px; max-height: 200px; }
            li { font-size: 14px; padding: 8px 10px; }
        }
    </style>
</head>
<body>
<div id="config-container" style="max-width: 960px; margin: 20px auto; padding:0 20px;">
    <h2>⚙️ Camera Configuration</h2>
    <input type="text" id="cameraIdInput" placeholder="Enter Camera ID" style="margin-right:10px; padding:5px;">
    <input type="text" id="stationNumberInput" placeholder="Enter Station Number" style="margin-right:10px; padding:5px;">
    <input type="text" id="customerIdInput" placeholder="Enter Customer ID" style="margin-right:10px; padding:5px;">
    <button id="saveConfigButton">✅ Save Config</button>
</div>
    <div id="video-container">
  <h2>🚀 SST Vision &mdash; Live Stream</h2>
  <div style="position:relative; display:inline-block;">
    <img id="stream" src="/stream" />
    <canvas id="roiCanvas"
      style="position:absolute; left:0; top:0; cursor:crosshair;"></canvas>
  </div>
  <button id="saveRoisButton" style="margin-top:10px;">✅ Save ROIs</button>
</div>
    <div class="box-wrapper">
        <div class="box">
            <h4>📌 Detected Events</h4>
            <ul id="event-list"></ul>
        </div>
        <div class="box">
            <h4>📈 Inference Log</h4>
            <ul id="inference-list"></ul>
        </div>
    </div>

    <script>
const canvas = document.getElementById('roiCanvas');
const img = document.getElementById('stream');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let startX, startY;
const rectangles = [];

function resizeCanvas() {
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    // Redraw all existing rectangles after resize
    drawAll();
}

// Wait for image to load before setting up canvas
img.onload = function() {
    resizeCanvas();
};

// Handle window resize
window.addEventListener('resize', resizeCanvas);

// Also handle image size changes (when stream updates)
img.addEventListener('load', resizeCanvas);

// Mouse handlers
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const currX = (e.clientX - rect.left) * scaleX;
    const currY = (e.clientY - rect.top) * scaleY;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAll();
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, currX - startX, currY - startY);
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;
    
    const label = prompt("Enter ROI Label:", `ROI ${rectangles.length+1}`);
    if (label) {
        // Calculate scaling factors to convert canvas coordinates to actual frame coordinates
        const frameScaleX = 640 / canvas.width;  // Assuming default frame width
        const frameScaleY = 480 / canvas.height; // Assuming default frame height
        
        // Get actual frame dimensions from server
        fetch('/frame_dimensions')
            .then(response => response.json())
            .then(frameDims => {
                const actualFrameScaleX = frameDims.width / canvas.width;
                const actualFrameScaleY = frameDims.height / canvas.height;
                
                rectangles.push({
                    label: label,
                    x1: Math.min(startX, endX) * actualFrameScaleX,
                    y1: Math.min(startY, endY) * actualFrameScaleY,
                    x2: Math.max(startX, endX) * actualFrameScaleX,
                    y2: Math.max(startY, endY) * actualFrameScaleY,
                    // Store canvas coordinates for display
                    canvas_x1: Math.min(startX, endX),
                    canvas_y1: Math.min(startY, endY),
                    canvas_x2: Math.max(startX, endX),
                    canvas_y2: Math.max(startY, endY)
                });
                
                isDrawing = false;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                drawAll();
            })
            .catch(err => {
                console.error('Error getting frame dimensions:', err);
                // Fallback to default scaling
                rectangles.push({
                    label: label,
                    x1: Math.min(startX, endX),
                    y1: Math.min(startY, endY),
                    x2: Math.max(startX, endX),
                    y2: Math.max(startY, endY),
                    canvas_x1: Math.min(startX, endX),
                    canvas_y1: Math.min(startY, endY),
                    canvas_x2: Math.max(startX, endX),
                    canvas_y2: Math.max(startY, endY)
                });
                
                isDrawing = false;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                drawAll();
            });
    } else {
        isDrawing = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawAll();
    }
});

function drawAll() {
    rectangles.forEach(r => {
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        // Use canvas coordinates for drawing
        const x1 = r.canvas_x1 || r.x1;
        const y1 = r.canvas_y1 || r.y1;
        const x2 = r.canvas_x2 || r.x2;
        const y2 = r.canvas_y2 || r.y2;
        
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.font = '16px Roboto';
        ctx.fillStyle = '#00FF00';
        ctx.fillText(r.label, x1 + 4, y1 - 4);
    });
}

document.getElementById('saveRoisButton').addEventListener('click', () => {
    if (rectangles.length === 0) {
        alert("No ROIs to save.");
        return;
    }
    
    // Send the actual frame coordinates (not canvas coordinates)
    const roisToSend = rectangles.map(r => ({
        label: r.label,
        x1: r.x1,
        y1: r.y1,
        x2: r.x2,
        y2: r.y2
    }));
    
    fetch('/update_rois', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rois: roisToSend })
    })
    .then(res => res.json())
    .then(data => {
        alert("ROIs saved successfully!");
        console.log(data);
        // Clear rectangles array so no duplicates
        rectangles.length = 0;
        // Clear the canvas so the old boxes disappear
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    })
    .catch(err => {
        console.error(err);
        alert("Error saving ROIs.");
    });
});

// Auto-refresh events and inference logs
setInterval(() => {
    fetch('/events_json')
        .then(response => response.json())
        .then(events => {
            const eventList = document.getElementById('event-list');
            eventList.innerHTML = '';
            events.forEach(event => {
                const li = document.createElement('li');
                li.textContent = event;
                li.className = event.includes('ALERT') ? 'red-dot' : 'green-dot';
                eventList.appendChild(li);
            });
        })
        .catch(err => console.error('Error fetching events:', err));

    fetch('/inference_json')
        .then(response => response.json())
        .then(inferences => {
            const inferenceList = document.getElementById('inference-list');
            inferenceList.innerHTML = '';
            inferences.forEach(inference => {
                const li = document.createElement('li');
                li.textContent = inference;
                li.className = 'yellow-dot';
                inferenceList.appendChild(li);
            });
        })
        .catch(err => console.error('Error fetching inference log:', err));
}, 1000);

document.getElementById('saveConfigButton').addEventListener('click', () => {
    const cameraId = document.getElementById('cameraIdInput').value.trim();
    const stationNumber = document.getElementById('stationNumberInput').value.trim();
    const customerId = document.getElementById('customerIdInput').value.trim();

    if (!cameraId || !stationNumber || !customerId) {
        alert("Please enter Camera ID, Station Number, and Customer ID.");
        return;
    }

    fetch('/update_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            camera_id: cameraId,
            station_number: stationNumber,
            customer_id: customerId
        })
    })
    .then(res => res.json())
    .then(data => {
        alert("Configuration saved successfully!");
        console.log(data);
    })
    .catch(err => {
        console.error(err);
        alert("Error saving configuration.");
    });
});

</script>
</body>
</html>
    ''')

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    camera_config["camera_id"] = data.get("camera_id", "CAM1")
    camera_config["station_number"] = data.get("station_number", "Station1")
    camera_config["customer_id"] = data.get("customer_id", "Customer1")
    print("[INFO] Updated camera config:", camera_config)
    return jsonify({"status": "success", "config": camera_config})

@app.route('/frame_dimensions')
def get_frame_dimensions():
    """Return the current frame dimensions for coordinate scaling"""
    return jsonify(frame_dimensions)

@app.route('/events_json')
def events_json():
    """Return events as JSON for AJAX updates"""
    return jsonify(event_log)

@app.route('/inference_json')
def inference_json():
    """Return inference log as JSON for AJAX updates"""
    return jsonify(inference_log)

@app.route('/events')
def events():
    def event_stream():
        last_len = 0
        while True:
            time.sleep(1)
            if len(event_log) > last_len:
                for msg in event_log[last_len:]:
                    yield f"data: {msg}\n\n"
                last_len = len(event_log)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route('/inference')
def inference():
    def inference_stream():
        last_len = 0
        while True:
            time.sleep(0.5)
            if len(inference_log) > last_len:
                for msg in inference_log[last_len:]:
                    yield f"data: {msg}\n\n"
                last_len = len(inference_log)
    return Response(stream_with_context(inference_stream()), mimetype="text/event-stream")

@app.route('/update_rois', methods=['POST'])
def update_rois():
    """
    Receive ROI definitions from the frontend as JSON and update the ROIs dictionary.
    Each call adds new ROIs instead of replacing all.
    """
    global ROIs
    data = request.json
    new_rois = {}
    for roi in data.get("rois", []):
        label = roi["label"]
        # Ensure coordinates are integers and properly ordered
        x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
        new_rois[label] = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]

    ROIs.update(new_rois)  # Merge with existing ROIs instead of replacing
    print("[INFO] Updated ROIs:", ROIs)
    return jsonify({"status": "success", "rois": ROIs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
