# Initial version
# import cv2
# from ultralytics import YOLO
#
#
# class StudentDetector:
#     def __init__(self, model_path="models/student_detector/yolov8n.pt", conf=0.45):
#         """Loads YOLO student detection model."""
#         self.model = YOLO(model_path)
#         self.conf = conf
#
#     def detect_students(self, frame):
#         """Runs YOLO detection and returns bounding boxes of students."""
#
#         results = self.model.predict(frame, conf=self.conf, verbose=False)
#
#         student_boxes = []
#
#         for r in results:
#             for box in r.boxes:
#
#                 cls_id = int(box.cls[0])
#                 class_name = self.model.names[cls_id]
#
#                 if class_name == "person":  # we only want students
#                     x1, y1, x2, y2 = box.xyxy[0].tolist()
#                     student_boxes.append({
#                         "bbox": (int(x1), int(y1), int(x2), int(y2)),
#                         "confidence": float(box.conf[0])
#                     })
#
#         return student_boxes
#
#     def draw_boxes(self, frame, boxes):
#         """Draws bounding boxes around detected students."""
#         for b in boxes:
#             (x1, y1, x2, y2) = b["bbox"]
#             conf = b["confidence"]
#
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"Student {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#         return frame

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Fasih Version
# module-6
# import cv2
# import numpy as np
# from ultralytics import YOLO
#
# class StudentDetector:
#     def __init__(self,
#                  model_path="yolov8n.pt",
#                  conf_threshold=0.40,
#                  person_class_id=0,
#                  auto_resize=True):
#         """
#         A universal student detection module compatible with:
#         YOLOv5, YOLOv8, YOLOv9, YOLO11, custom models.
#
#         Parameters:
#         --------------------------------------------------
#         model_path        : Path to YOLO model file (.pt)
#         conf_threshold    : Minimum confidence for detection
#         person_class_id   : Class ID for "person"
#         auto_resize       : Auto-handle non-standard image shapes
#         """
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold
#         self.person_class_id = person_class_id
#         self.auto_resize = auto_resize
# #Output (yolo)
#     def _safe_extract_box(self, box):
#         """Safely extract xyxy as int for any YOLO output format."""
#         try:
#             xyxy = box.xyxy[0].cpu().numpy().astype(int)
#         except:
#             try:
#                 xyxy = np.array(box.xyxy).astype(int)
#             except:
#                 return None
#         return xyxy.tolist()
# #Detection
#     def detect_students(self, frame):
#         """
#         Input  -> frame (numpy array)
#         Output -> list of [x1, y1, x2, y2] bounding boxes
#         """
#         if frame is None:
#             return []
#
#         if self.auto_resize and frame.shape[0] < 10:
#             return []
#
#         try:
#             results = self.model(frame, verbose=False)[0]
#         except Exception as e:
#             print("❌ YOLO inference error:", e)
#             return []
#
#         student_boxes = []
#
#         for box in results.boxes:
#             try:
#                 cls = int(box.cls)
#                 conf = float(box.conf)
#             except:
#                 continue
#
#             if cls == self.person_class_id and conf >= self.conf_threshold:
#                 xyxy = self._safe_extract_box(box)
#                 if xyxy:
#                     student_boxes.append(xyxy)
#
#         return student_boxes
# #debugging
#     def draw_students(self, frame, student_boxes):
#         if frame is None:
#             return frame
#
#         for (x1, y1, x2, y2) in student_boxes:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, "Student", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                         (0, 255, 0), 2)
#         return frame
# #Optional SELF-TEST
# if __name__ == "__main__":
#     detector = StudentDetector("yolov8n.pt")
#
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("❌ ERROR: Cannot open webcam.")
#         exit()
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Frame error")
#             continue
#
#         boxes = detector.detect_students(frame)
#         frame_out = detector.draw_students(frame, boxes)
#
#         cv2.imshow("Module 6 - Student Detection (Flexible)", frame_out)
#
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Trained Model
# from ultralytics import YOLO
# import cv2
#
# class StudentDetector:
#     def __init__(self, model_path="C:\\Users\\khana\\PycharmProjects\\AI_Mobile_Detector\\models\\student_detector\\best.pt"):
#         # Load trained YOLO model
#         self.model = YOLO(model_path)
#
#     def detect_students(self, image_path="D:\Gallery\Friends\WhatsApp Image 2024-12-14 at 14.23.19_4e89ba06.jpg"):
#         # Load the image
#         image = cv2.imread(image_path)
#
#         if image is None:
#             raise FileNotFoundError(f"Could not read the image: {image_path}")
#
#         # Run YOLO inference
#         results = self.model(image)
#
#         student_count = 0
#
#         # Parse detection results
#         for r in results:
#             for box in r.boxes:
#                 cls = int(box.cls[0])      # class id
#                 conf = float(box.conf[0])  # confidence
#
#                 # Assuming your dataset has one class: "student"
#                 if cls == 0:  # class 0 is student
#                     student_count += 1
#
#                     # Draw box on image
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(image, f"Student {conf:.2f}",
#                                 (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.6,
#                                 (0, 255, 0),
#                                 2)
#
#         # Show result image
#         cv2.imshow("Student Detection", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         return student_count
#
#
# # -------------------------
# # Example usage
# # -------------------------
# if __name__ == "__main__":
#     detector = StudentDetector("C:\\Users\\khana\\PycharmProjects\\AI_Mobile_Detector\\models\\student_detector\\best.pt")
#
#     image_path = "D:\Gallery\Friends\WhatsApp Image 2024-12-14 at 14.23.19_4e89ba06.jpg"   # change this to your image
#     count = detector.detect_students(image_path)
#
#     print(f"Total students detected: {count}")

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# import cv2
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from ultralytics import YOLO
# from utils.preprocessing_module9 import VideoProcessor
#
# class StudentDetectorVideo:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#
#     def detect_frame(self, frame):
#         """
#         Detect students in a single frame.
#         """
#         # Make sure input is uint8 RGB
#         if frame.dtype != 'uint8':
#             frame = (frame * 255).astype('uint8')
#
#         results = self.model(frame)  # YOLOv8 inference
#
#         student_count = 0
#
#         # Loop through detections
#         for result in results:
#             boxes = result.boxes
#             if boxes is None:
#                 continue
#             for box in boxes:
#                 cls = int(box.cls[0])
#                 conf = float(box.conf[0])
#                 # Make sure class 0 corresponds to your student class
#                 if cls == 0:
#                     student_count += 1
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"Student {conf:.2f}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         return frame, student_count
#
#
# # =====================
# # Real-time Video Test
# # =====================
# if __name__ == "__main__":
#     processor = VideoProcessor(threaded=True, output_format="rgb")  # RGB for YOLO
#     processor.capture_video(0)  # webcam
#
#     detector = StudentDetectorVideo(
#         model_path="C:\\Users\\khana\\PycharmProjects\\AI_Mobile_Detector\\models\\student_detector\\best.pt"
#     )
#
#     while True:
#         ret, frame = processor.read_frame()
#         if not ret or frame is None:
#             continue
#
#         processed_frame = processor.preprocess(frame)  # resize + RGB
#         output_frame, count = detector.detect_frame(processed_frame)
#
#         # Show FPS
#         cv2.putText(output_frame, f"FPS: {processor.fps:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         cv2.imshow("Student Detection", output_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('0'):
#             break
#
#     processor.stop()
#     cv2.destroyAllWindows()

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Khadija Version
# from ultralytics import YOLO     # YOLOv8 model for person detection and tracking
# import cv2                      # OpenCV for image/video processing and drawing
#
# # ============================================================
# # Module 6 – Student Detection & Tracking
# # Integration-ready for other modules (e.g., phone detection)
# # ============================================================
#
# class StudentDetector:
#     def __init__(self, model_path="yolov8n.pt", conf_threshold=0.6, person_class_id=0):
#         """
#         Initialize YOLO model for detecting students (persons).
#
#         Why YOLO?
#         - Deep learning model capable of detecting objects in real-time.
#         - Built-in tracking assigns unique IDs.
#         - High accuracy and good speed.
#
#         Parameters:
#         - model_path: path to YOLOv8 model file.
#         - conf_threshold: minimum confidence to accept a detection.
#         - person_class_id: COCO class ID for 'person' (0 = person).
#         """
#
#         self.model = YOLO(model_path)      # Load YOLOv8 model
#         self.conf_threshold = conf_threshold
#         self.person_class_id = person_class_id
#
#     # ------------------------------------------------------------
#     # Process each frame: detect persons, track them, draw boxes
#     # ------------------------------------------------------------
#     def process_frame(self, frame, students_with_phone=None):
#         """
#         Main function that:
#         - Runs YOLO detection and tracking
#         - Extracts student coordinates & IDs
#         - Draws bounding boxes (green or red)
#         - Returns (processed_frame, student_data)
#
#         students_with_phone:
#         - Set of student IDs that have a phone detected by Module 4.
#         """
#
#         if frame is None:
#             return frame, []
#
#         # Resize frame for faster processing (optional)
#         frame = cv2.resize(frame, (640, 480))
#
#         # Run YOLO with tracking (only detect persons → class 0)
#         results = self.model.track(
#             frame,
#             conf=self.conf_threshold,     # confidence filter
#             persist=True,                 # enables tracking IDs
#             classes=[self.person_class_id]
#         )
#
#         student_data = []                # stores student info for integration
#         students_with_phone = students_with_phone or set()
#
#         # Parse detection results
#         if results is not None:
#             for r in results:
#                 if r.boxes is None:
#                     continue
#
#                 # Loop through detected bounding boxes
#                 for box in r.boxes:
#                     if box.id is None:
#                         continue   # skip if YOLO couldn't assign ID
#
#                     # Extract coordinates and values
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     conf = float(box.conf[0])
#                     track_id = int(box.id[0])    # student unique ID
#
#                     # Store student information
#                     student_info = {
#                         "id": track_id,
#                         "x1": int(x1),
#                         "y1": int(y1),
#                         "x2": int(x2),
#                         "y2": int(y2),
#                         "conf": conf
#                     }
#                     student_data.append(student_info)
#
#                     # -------------------------
#                     # Drawing logic
#                     # -------------------------
#                     color = (0, 255, 0)         # green = normal student
#                     label = f"Student {track_id}"
#
#                     # If phone detected in another module → highlight in red
#                     if track_id in students_with_phone:
#                         color = (0, 0, 255)     # red = warning
#                         label = f"Student {track_id} (Phone!)"
#
#                     # Draw bounding box around student
#                     cv2.rectangle(
#                         frame,
#                         (int(x1), int(y1)),
#                         (int(x2), int(y2)),
#                         color,
#                         2
#                     )
#
#                     # Draw label (Student ID)
#                     cv2.putText(
#                         frame,
#                         label,
#                         (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7,
#                         color,
#                         2
#                     )
#
#         # Display total student count on screen
#         num_students = len(student_data)
#         cv2.putText(
#             frame,
#             f"Total Students: {num_students}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )
#
#         return frame, student_data
#
#
# # ============================================================
# # Integration Hook: Phone Detection Module Placeholder
# # ============================================================
#
# def get_students_with_phone(student_data, frame):
#     """
#     This function is designed as an integration point.
#     Why?
#     - Module 4 (Phone Detection) will use these coordinates
#     - It will decide which student has a mobile phone.
#
#     student_data: list of student bounding boxes
#     frame: full camera frame
#
#     Currently: returns empty set (no students have phones).
#     """
#
#     # Replace with real phone detection logic later
#     return set()
#
#
# # ============================================================
# # Main runner for webcam, video, or image
# # ============================================================
#
# def run_module_6(
#     use_webcam=True,
#     video_path="students_video.mp4",
#     image_path="classroom.jpg"
# ):
#     """
#     Start the system in:
#     - Webcam mode
#     - Video mode
#     - Image mode
#
#     Handles:
#     ✔ Reading frames
#     ✔ Sending frames to student detection
#     ✔ Integrating with phone detection
#     ✔ Displaying results
#     """
#
#     detector = StudentDetector()     # Create student detector object
#
#     # ------------- Select Input Source -------------
#     if use_webcam is True:
#         cap = cv2.VideoCapture(0)     # webcam stream
#     elif use_webcam is False:
#         cap = cv2.VideoCapture(video_path)  # video file
#     else:
#         # IMAGE MODE
#         frame = cv2.imread(image_path)
#         if frame is None:
#             print("Error: could not load image:", image_path)
#             return
#
#         # Detect students in image
#         students_with_phone = get_students_with_phone([], frame)
#         processed_frame, student_data = detector.process_frame(
#             frame,
#             students_with_phone=students_with_phone
#         )
#
#         # Show output
#         cv2.imshow("Module 6 - Student Detection (Image)", processed_frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         print("Detected students data (image):", student_data)
#         return
#
#     # ------------- Video / Webcam Loop -------------
#     all_frames_data = []
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("End of stream or cannot read frame.")
#             break
#
#         # Ask phone module if any student has phone
#         students_with_phone = get_students_with_phone([], frame)
#
#         # Detect students + draw bounding boxes
#         processed_frame, student_data = detector.process_frame(
#             frame,
#             students_with_phone=students_with_phone
#         )
#
#         all_frames_data.append(student_data)
#
#         # Display live output
#         cv2.imshow("Module 6 - Student Detection (Video/Webcam)", processed_frame)
#
#         # Quit using 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release after exit
#     cap.release()
#     cv2.destroyAllWindows()
#
#     print("Detected students data (all frames):")
#     print(all_frames_data)
#
#
# # ============================================================
# # Script Entry Point
# # ============================================================
#
# if __name__ == "__main__":
#     run_module_6(
#         use_webcam=True,               # True = Webcam, False = Video, None = Image
#         video_path="students_video.mp4",
#         image_path="classroom.jpg"
#     )
#     """
#             Module 6: Detects students (persons), assigns IDs, and provides coordinates.
#             This module is designed to be reusable and easy to integrate with other modules.
#
#             Functional Requirements:
#             - Detect all students using a person-detection deep learning model (YOLOv8).
#             - Draw bounding boxes around detected students.
#             - Assign unique IDs to each detected student.
#             - Store coordinates for integration with other modules.
#             - Support real-time and static image processing.
#             - Filter non-student objects.
#             """
#     #press 'q' to quit

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Integrated Version Yolo Based

# ============================
# Integrated Module 6 + Module 9
# ============================

import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing_module9 import VideoProcessor  # Module 9
from ultralytics import YOLO

# ------------------------------
# Student Detector (Module 6)
# ------------------------------
class StudentDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.6, person_class_id=0):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

    def process_frame(self, frame, students_with_phone=None):
        if frame is None:
            return frame, []

        frame = cv2.resize(frame, (640, 480))
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            persist=True,
            classes=[self.person_class_id]
        )

        student_data = []
        students_with_phone = students_with_phone or set()

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0])
                student_info = {"id": track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf}
                student_data.append(student_info)

                color = (0, 255, 0)
                label = f"Student {track_id}"
                if track_id in students_with_phone:
                    color = (0, 0, 255)
                    label = f"Student {track_id} (Phone!)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Total Students: {len(student_data)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, student_data

# ------------------------------
# Placeholder for phone detection
# ------------------------------
def get_students_with_phone(student_data, frame):
    return set()  # replace with real logic later

# ------------------------------
# Integrated runner
# ------------------------------
def run_student_detection_with_preprocessing(use_webcam=True, video_path="students_video.mp4"):
    # Initialize video processor (Module 9)
    processor = VideoProcessor(threaded=True, output_format="rgb")  # RGB uint8 for YOLO
    if use_webcam:
        processor.capture_video(0)
    else:
        processor.capture_video(video_path)

    # Initialize student detector (Module 6)
    detector = StudentDetector(model_path="yolov8n.pt")

    all_frames_data = []

    while True:
        ret, frame = processor.read_frame()
        if not ret or frame is None:
            continue

        # Preprocess frame
        processed_frame = processor.preprocess(frame)  # resize + RGB (or other enhancements)

        # Detect students
        students_with_phone = get_students_with_phone([], processed_frame)
        output_frame, student_data = detector.process_frame(processed_frame, students_with_phone)

        all_frames_data.append(student_data)

        # Show FPS on frame
        cv2.putText(output_frame, f"FPS: {processor.fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Integrated Student Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.stop()
    cv2.destroyAllWindows()
    print("All frames detected student data:")
    print(all_frames_data)

# ------------------------------
# Run integrated system
# ------------------------------
# if __name__ == "__main__":
#     run_student_detection_with_preprocessing(use_webcam=True)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
# # Integrated Version MediaPipe Based
#
# import cv2
# import mediapipe as mp
# import numpy as np
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.preprocessing_module9 import VideoProcessor  # Module 9
#
#
# class StudentDetector:
#     """
#     =============================================================================
#                         MODULE 10 – STUDENT DETECTION (MEDIAPIPE)
#     =============================================================================
#     This module detects students using:
#         - Face landmarks
#         - Upper-body landmarks (shoulders)
#
#     OUTPUT:
#         A list of student detections:
#         [
#             {
#                 "id": 1,
#                 "bbox": (x1, y1, x2, y2),
#                 "center": (cx, cy)
#             },
#             ...
#         ]
#
#     =============================================================================
#     """
#
#     def __init__(self):
#         self.mp_holistic = mp.solutions.holistic
#         self.holistic = self.mp_holistic.Holistic(
#             static_image_mode=False,
#             model_complexity=1,
#             enable_segmentation=False,
#             refine_face_landmarks=True
#         )
#
#     def detect_students(self, rgb_frame):
#         """
#         Detect students using Mediapipe Holistic.
#         Returns list of bounding boxes.
#
#         Parameters:
#         -----------
#         rgb_frame : Preprocessed RGB frame from Module 9
#
#         Returns:
#             detections : list of dictionaries containing bounding boxes
#         """
#         results = self.holistic.process(rgb_frame)
#         height, width, _ = rgb_frame.shape
#
#         detections = []
#
#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#
#             # Shoulders (upper body anchor points)
#             LEFT_SH = lm[self.mp_holistic.PoseLandmark.LEFT_SHOULDER]
#             RIGHT_SH = lm[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER]
#
#             # Nose (head anchor)
#             NOSE = lm[self.mp_holistic.PoseLandmark.NOSE]
#
#             # Convert to pixel positions
#             x1 = int(min(LEFT_SH.x, RIGHT_SH.x) * width)
#             x2 = int(max(LEFT_SH.x, RIGHT_SH.x) * width)
#             y1 = int((NOSE.y - 0.1) * height)
#             y2 = int((LEFT_SH.y + 0.15) * height)
#
#             # Add to detections
#             detections.append({
#                 "bbox": (x1, y1, x2, y2),
#                 "center": ((x1 + x2)//2, (y1 + y2)//2)
#             })
#
#         return detections
#
#
# # =============================================================================
# #                           MAIN TESTING PIPELINE
# # =============================================================================
# def main():
#     # Use Module 9 VideoProcessor
#     processor = VideoProcessor(threaded=True, output_format="rgb")
#     processor.capture_video(0)
#
#     detector = StudentDetector()
#
#     while True:
#         ret, frame = processor.read_frame()
#         if not ret or frame is None:
#             continue
#
#         # Preprocess frame (RGB)
#         rgb_frame = processor.preprocess(frame)
#
#         # Detect students
#         detections = detector.detect_students(rgb_frame)
#
#         # Draw results
#         for det in detections:
#             x1, y1, x2, y2 = det["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cx, cy = det["center"]
#             cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
#
#         cv2.putText(frame, f"FPS: {processor.fps:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         cv2.imshow("Student Detector (Module 10)", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     processor.stop()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()