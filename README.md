# AI-Based Classroom Mobile Phone Detection System 

An end-to-end **computer vision system** designed to detect **students using mobile phones in a classroom environment**.  
The system integrates multiple AI detection modules and a central logic engine to **identify, associate, and log phone usage by individual students in real time**.

---

##  Project Overview

This project uses **YOLOv8**, **MediaPipe**, and **custom logic modules** to:

- Detect students in a classroom
- Detect mobile phones
- Associate phones to specific students
- Analyze supporting cues (hand presence, head pose, screen glow)
- Trigger alerts and log violations

The architecture is **modular**, **scalable**, and **real-time ready**.

---

## 🧠System Architecture

The system is divided into **independent detection modules** and a **central logic engine**.
**Camera/Video Processing**
  Videp capture(webcam)
  Frame resizing
  Color format handling(BGR/RGB)
  Threaded frame reading
**Student Detection**
  YOLOv8-based person detection
  Multi-object tracking with IDs
  Assigns unique student ID
  Outputs bounding boxes per student
**Phone Detection**
  YOLOv8 COCO model
  Detects cell phones (class ID 67)
  Returns bounding boxes + confidence
  Triggers audio alerts on detection
**Hand Detection**
  Detects hands near phones
  Optionally associates hand to students
  Improves phone ownership accuracy
**Head Pose Detection**
  Uses MediaPipe Face Mesh
  Estimates pitch and yaw angle
  Detects head-down behavior
**Screen Glow Detection**
  Detects bright regions inside phone bounding boxes
  Estimates screen activity using brightness analysis
  Adds confidence boost to phone usage detection
**Phone to Student Association**
  Associates each detected phone with the most likely student using:
    Intersection over Union
    Center distance
    Hand intersection
    Screen glow proximity
    Head pose orientation
**Student Status Logic Engine**
  Consumes outputs of all detection modules
  Determines finaal student status (suspect/normal/phone confirmed)
  Applies time-based vision
  Handles alert rules and logging
**Alerts and Logging**
  Audio alert when phone usage is confirmed
  Bounding box visualization
  Per-student violation logs
  Optional phone image cropping

**Tech Stack**
  Python 3.9+
  YOLOv8 (Ultralytics)
  OpenCV
  MediaPipe
  NumPy
  winsound

**Author**
Ahad Khan

**License**
This project is for educational and research purposes. Feel free to experiment and improve.
