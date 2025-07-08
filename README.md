# 🚀 Smart Video Analytics Platform

A Flask web application for real-time video analytics powered by **YOLOv8 object detection** and **Google Cloud Storage** integration.

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

---

### 🗺️ Configurable ROIs
- Draw ROIs in the browser
- Save them with custom labels
- Detection results filtered by ROI

---

### 🚨 Event Logging & Alerts
- Idle vehicle alerts (with adjustable thresholds)
- Unattended vehicle alerts
- Person using mobile phone alerts
- Logs with timestamps, saved frames, and alert levels

---

### ☁️ Cloud Storage Integration
- Snapshots of events uploaded to Google Cloud Storage

---

### ⚡ Dynamic Camera Configuration
- Update Camera ID, Station Number, and Customer ID via the UI

---

## 🏗️ Architecture

**Components:**

**Flask Backend**
- Streams RTSP video
- Runs YOLOv8 detection & tracking
- Processes event logic
- Provides REST API for config, events, and ROI management

**Frontend**
- HTML5/CSS3/JavaScript (single-page)
- Video stream with overlaid detections
- ROI drawing canvas
- Event logs and inference logs

**Google Cloud Storage**
- Stores annotated frames of important events

---

## ⚙️ Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd <your-project-directory>
