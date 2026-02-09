# # Integrated V1
#
# import cv2
# import threading
# import sys
# import os
# import time
# import numpy as np
# import pygame
#
# # =========================================
# # Add project root to path for imports
# # =========================================
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath('./detection'))
# sys.path.append(os.path.abspath('./utils'))
#
#
# # =========================================
# # Import Modules
# # =========================================
# from preprocessing_module9 import VideoProcessor
# from student_detector_module6 import StudentDetector
# from head_pose_module3 import HeadPoseDetector
# from phone_detector_module4 import PhoneDetector
# from screen_glow_module7 import detect_phones as screen_glow_detect
# from hand_detector_module5 import HandDetector
#
# # =========================================
# # Initialize Pygame for alert sounds
# # =========================================
# pygame.mixer.init()
# ALERT_SOUND_PATH = "C:\\Users\\khana\\OneDrive\\Desktop\\WhatsApp Ptt 2025-12-06 at 20.08.49.ogg"  # replace with your OGG file path
# try:
#     alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
# except Exception as e:
#     print(f"[WARN] Could not load alert sound: {e}")
#     alert_sound = None
#
# def play_alert():
#     if alert_sound:
#         try:
#             alert_sound.play()
#         except:
#             pass
#
# # =========================================
# # Main integration
# # =========================================
# def main(video_source=0):
#     # Initialize video processor
#     processor = VideoProcessor(threaded=True, output_format="bgr")
#     processor.capture_video(video_source)
#
#     # Initialize modules
#     student_detector = StudentDetector()
#     head_pose_detector = HeadPoseDetector()
#     phone_detector = PhoneDetector()
#     hand_detector = HandDetector(max_hands=6)
#
#     print("[INFO] Starting integrated detection. Press 'q' to quit.")
#
#     while True:
#         ret, frame = processor.read_frame()
#         if not ret or frame is None:
#             continue
#
#         # Preprocess frame
#         processed_frame = processor.preprocess(frame)
#
#         # ------------------------
#         # Phone detection
#         # ------------------------
#         phone_detections = phone_detector.detect(processed_frame)
#         if phone_detections and alert_sound:
#             threading.Thread(target=play_alert, daemon=True).start()
#         output_frame = phone_detector.draw(processed_frame.copy(), phone_detections)
#
#         # ------------------------
#         # Screen glow detection (additional)
#         # ------------------------
#         output_frame, _ = screen_glow_detect(output_frame)
#
#         # ------------------------
#         # Student detection
#         # ------------------------
#         students_with_phone = {det["id"] for det in phone_detections}  # naive linking
#         output_frame, student_data = student_detector.process_frame(output_frame, students_with_phone)
#
#         # ------------------------
#         # Head pose detection
#         # ------------------------
#         output_frame, head_poses = head_pose_detector.process_frame(output_frame)
#
#         # ------------------------
#         # Hand detection
#         # ------------------------
#         output_frame, hands_data = hand_detector.process_frame(output_frame)
#
#         # ------------------------
#         # Display FPS
#         # ------------------------
#         cv2.putText(output_frame, f"FPS: {processor.fps:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#
#         # Show output
#         cv2.imshow("Integrated Detection", output_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     processor.stop()
#     cv2.destroyAllWindows()
#
#
# # =========================================
# # Entry Point
# # =========================================
# if __name__ == "__main__":
#     main(video_source=0)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Integrated V2
#
import cv2
import time
import sys
import os
import winsound
import threading
import numpy as np

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath('./detection'))
sys.path.append(os.path.abspath('./utils'))
sys.path.append(os.path.abspath('./logic'))

# --------------------------------------------------
# IMPORT MODULES
# --------------------------------------------------
from utils.preprocessing_module9 import VideoProcessor
from detection.hand_detector_module5 import HandDetector
from detection.head_pose_module3 import HeadPoseDetector
from detection.phone_detector_module4 import PhoneDetector
from detection.screen_glow_module7 import ScreenGlowDetector as screen_glow_detect, ScreenGlowDetector
from logic.activity_classifier_module1 import ActivityClassifier

# --------------------------------------------------
# STUDENT DETECTOR (MODULE 6)
# --------------------------------------------------
from ultralytics import YOLO

class StudentDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.6):
        self.model = YOLO(model_path)
        self.conf = conf

    def process(self, frame, students_with_phone):
        students = []

        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=[0]  # person
        )

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                sid = int(box.id[0])

                students.append({
                    "id": sid,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                color = (0, 255, 0)
                label = f"Student {sid}"

                if sid in students_with_phone:
                    color = (0, 0, 255)
                    label += " (PHONE)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, students


# --------------------------------------------------
# ALERT FUNCTION
# --------------------------------------------------
def play_alert():
    try:
        winsound.Beep(1000, 200)
    except:
        pass


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
def main():

    # Video capture
    processor = VideoProcessor(threaded=True, output_format="bgr")
    processor.capture_video(0)

    # Initialize modules
    student_detector = StudentDetector()
    hand_detector = HandDetector()
    head_detector = HeadPoseDetector()
    phone_detector = PhoneDetector()
    activity_classifier = ActivityClassifier(screen_glow_threshold=50)

    last_alert = {}
    ALERT_COOLDOWN = 2.0

    print("[INFO] System started. Press 'q' to quit.")

    while True:
        ret, frame = processor.read_frame()
        if not ret or frame is None:
            continue

        frame = processor.preprocess(frame)

        # -----------------------------
        # 1. Phone Detection (Module 4)
        # -----------------------------
        phone_detections = phone_detector.detect(frame)
        frame = phone_detector.draw(frame, phone_detections)

        # -----------------------------
        # 2. Screen Glow Detection (Module 7)
        # -----------------------------
        screen_glow_detector = ScreenGlowDetector(model_path="yolov8n.pt")
        frame, screen_glow_detected = screen_glow_detector.detect(frame)

        # Add brightness info to phone_detections for activity classifier
        # (if screen_glow_detected returns bounding boxes, attach them)
        for pd in phone_detections:
            pd["brightness"] = 100 if screen_glow_detected else 0  # placeholder

        # -----------------------------
        # 3. Hand Detection (Module 5)
        # -----------------------------
        frame, hands_data = hand_detector.process_frame(frame)

        # -----------------------------
        # 4. Head Pose Detection (Module 3)
        # -----------------------------
        frame, head_poses = head_detector.process_frame(frame)

        # -----------------------------
        # 5. Student Detection (Module 6)
        # -----------------------------
        students_with_phone_ids = {i for i in range(len(phone_detections))}  # naive mapping
        frame, student_data = student_detector.process(frame, students_with_phone_ids)

        # -----------------------------
        # 6. Activity Classification (Module 1)
        # -----------------------------
        activity_results = activity_classifier.classify(
            student_data=student_data,
            phone_detections=phone_detections,
            hands_data=hands_data,
            head_poses=head_poses
        )

        frame = activity_classifier.visualize(frame, student_data, activity_results, phone_detections)

        # -----------------------------
        # 7. ALERT SYSTEM
        # -----------------------------
        now = time.time()
        for sid, activity in activity_results.items():
            if activity == "Actively Using Phone":
                last = last_alert.get(sid, 0)
                if now - last > ALERT_COOLDOWN:
                    threading.Thread(target=play_alert, daemon=True).start()
                    last_alert[sid] = now

        # -----------------------------
        # 8. DISPLAY
        # -----------------------------
        cv2.putText(frame, f"FPS: {processor.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("AI Mobile Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.stop()
    cv2.destroyAllWindows()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Integrated V3

# import cv2
# import threading
# import time
# import sys
# import os
# import winsound
#
# # -----------------------------
# # PATH SETUP
# # -----------------------------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# # sys.path.append(os.path.abspath('./detection'))
# # sys.path.append(os.path.abspath('./utils'))
# # sys.path.append(os.path.abspath('./logic'))
#
# # -----------------------------
# # IMPORT MODULES
# # -----------------------------
# from utils.preprocessing_module9 import VideoProcessor
# from detection.student_detector_module6 import StudentDetector
# from detection.head_pose_module3 import HeadPoseDetector
# from detection.phone_detector_module4 import PhoneDetector
# from detection.screen_glow_module7 import ScreenGlowDetector
# from detection.hand_detector_module5 import HandDetector
# from logic.activity_classifier_module1 import ActivityClassifier
# from logic.phone_student_association_module2 import associate_phones_to_students
#
#
# # -----------------------------
# # ALERT FUNCTION
# # -----------------------------
# def play_alert():
#     winsound.Beep(1000, 200)
#
# # -----------------------------
# # INTEGRATED SYSTEM CLASS
# # -----------------------------
# class AIMobileDetectionSystem:
#     def __init__(self, video_source=0):
#         # Video
#         self.processor = VideoProcessor(threaded=True, output_format="bgr")
#         self.processor.capture_video(video_source)
#
#         # Detectors
#         self.student_detector = StudentDetector(model_path="yolov8n.pt")
#         self.head_detector = HeadPoseDetector()
#         self.phone_detector = PhoneDetector()
#         self.screen_glow_detector = ScreenGlowDetector(model_path="yolov8n.pt")
#         self.hand_detector = HandDetector(max_hands=6)
#         self.activity_classifier = ActivityClassifier()  # Module 1
#
#         # Tracking
#         self.last_alert = {}
#         self.alert_cooldown = 2.0  # seconds
#
#     def run(self):
#         print("[INFO] System started. Press 'q' to quit.")
#
#         while True:
#             ret, frame = self.processor.read_frame()
#             if not ret or frame is None:
#                 continue
#
#             # Preprocess frame
#             frame = self.processor.preprocess(frame)
#
#             # ------------------------
#             # 1. Phone Detection
#             # ------------------------
#             phone_detections = self.phone_detector.detect(frame)
#             if phone_detections:
#                 threading.Thread(target=play_alert, daemon=True).start()
#             frame = self.phone_detector.draw(frame.copy(), phone_detections)
#
#             # ------------------------
#             # 2. Screen Glow Detection
#             # ------------------------
#             frame, screen_glow_detected = self.screen_glow_detector.detect(frame)
#             if screen_glow_detected:
#                 threading.Thread(target=play_alert, daemon=True).start()
#
#             # ------------------------
#             # 3. Hand Detection
#             # ------------------------
#             frame, hand_data = self.hand_detector.process_frame(frame)
#
#             # ------------------------
#             # 4. Head Pose Detection
#             # ------------------------
#             frame, head_pose_data = self.head_detector.process_frame(frame)
#
#             # ------------------------
#             # 6. Student Detection
#             # ------------------------
#             # students_with_phone will be updated by Module 2 association later
#             students_with_phone = set()
#             frame, student_data = self.student_detector.process_frame(frame, students_with_phone)
#
#             # ------------------------
#             # 5. Phone to Student Association
#             # ------------------------
#             students_with_phone, associations = associate_phones_to_students(student_boxes=student_data,
#                                                                              phone_boxes=phone_detections,
#                                                                              hand_boxes=hand_data,
#                                                                              head_pose=head_pose_data,
#                                                                              screen_glow=screen_glow_detected)
#
#             # ------------------------
#             # 7. Activity Classification (Module 1)
#             # ------------------------
#             # Provide all inputs to classifier
#             classified = self.activity_classifier.classify(
#                 student_data=student_data,
#                 phone_detections=phone_detections,
#                 hands_data=hand_data,
#                 head_poses=head_pose_data,
#             )
#
#             # Display classification labels on frame
#             for item in classified:
#                 sid = item['student_id']
#                 label = item['activity']
#                 x1, y1 = item['x1'], item['y1']
#                 cv2.putText(frame, label, (x1, y1-20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#
#             # ------------------------
#             # Display FPS
#             # ------------------------
#             cv2.putText(frame, f"FPS: {self.processor.fps:.1f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#             # Show frame
#             cv2.imshow("AI Mobile Detection System", frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         # Cleanup
#         self.processor.stop()
#         cv2.destroyAllWindows()
#
# # -----------------------------
# # ENTRY POINT
# # -----------------------------
# if __name__ == "__main__":
#     system = AIMobileDetectionSystem(video_source=0)
#     system.run()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Integrated V3

# import cv2
# import sys
# import os
# import time
# import threading
# import winsound  # for alert beep
# from torchvision.transforms.v2 import Lambda
#
# # -------------------------------
# # PATH SETUP
# # -------------------------------
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath('./detection'))
# sys.path.append(os.path.abspath('./utils'))
# sys.path.append(os.path.abspath('./logic'))
#
# # -------------------------------
# # IMPORT MODULES
# # -------------------------------
# from utils.preprocessing_module9 import VideoProcessor
# from detection.student_detector_module6 import StudentDetector
# from detection.phone_detector_module4 import PhoneDetector
# from detection.screen_glow_module7 import ScreenGlowDetector
# from detection.hand_detector_module5 import HandDetector
# from detection.head_pose_module3 import HeadPoseDetector
# from logic.phone_student_association_module2 import associate_phones_to_students
# from logic.activity_classifier_module1 import ActivityClassifier
# from logic.status_engine_module8 import StudentStatusLogicEngine
#
#
# # -------------------------------
# # ALERT FUNCTION
# # -------------------------------
# # ALERT_COOLDOWN = 2.0
# # last_alert = {}
# #
# # def play_alert():
# #     winsound.Beep(1000, 250)  # 1kHz, 250ms
# import queue
# ALERT_COOLDOWN = 2.0
# last_alert = {}
# alert_queue = queue.Queue()
#
# def alert_worker():
#     while True:
#         sid =  alert_queue.get()
#         winsound.Beep(1000,250)
#         alert_queue.task_done()
# threading.Thread(target=alert_worker,daemon=True).start()
#
#
# # -------------------------------
# # INTEGRATED DETECTION SYSTEM CLASS
# # -------------------------------
# class DetectionSystem:
#     def __init__(self, video_source=0):
#         self.processor = VideoProcessor(threaded=True, output_format="bgr")
#         self.video_source = video_source
#
#         # Initialize modules
#         self.student_detector = StudentDetector()
#         self.phone_detector = PhoneDetector()
#         self.screen_glow_detector = ScreenGlowDetector()
#         self.hand_detector = HandDetector(max_hands=6)
#         self.head_pose_detector = HeadPoseDetector()
#         self.activity_classifier = ActivityClassifier()
#         self.status_engine = StudentStatusLogicEngine()
#
#     def run(self):
#         self.processor.capture_video(self.video_source)
#         print("[INFO] System started. Press 'q' to quit.")
#
#         while True:
#             ret, frame = self.processor.read_frame()
#             if not ret or frame is None:
#                 continue
#
#             # Preprocess frame
#             frame = self.processor.preprocess(frame)
#
#             # -----------------------------
#             # 1. Phone Detection
#             # -----------------------------
#             phone_detections = self.phone_detector.detect(frame)
#
#             # -----------------------------
#             # 2. Screen Glow Detection
#             # -----------------------------
#             frame, screen_glow_detected = self.screen_glow_detector.detect(frame)
#
#             # -----------------------------
#             # 3. Hand Detection
#             # -----------------------------
#             frame, hand_data = self.hand_detector.process_frame(frame)
#
#             # -----------------------------
#             # 4. Head Pose Detection
#             # -----------------------------
#             frame, head_pose_data = self.head_pose_detector.process_frame(frame)
#
#             # -----------------------------
#             # 5. Student Detection
#             # -----------------------------
#             frame, student_data = self.student_detector.process_frame(frame)
#
#             # -----------------------------
#             # 6. Phone-to-Student Association
#             # -----------------------------
#             students_with_phone_ids, associations = associate_phones_to_students(
#                 student_boxes=student_data,
#                 phone_boxes=phone_detections,
#                 hand_boxes=hand_data,
#                 head_pose=head_pose_data
#             )
#
#             # -----------------------------
#             # 7. Student Status Logic Engine
#             # -----------------------------
#             student_statuses = self.status_engine.update(
#                 students=student_data,
#                 students_with_phone=students_with_phone_ids,
#                 head_pose_data=head_pose_data,
#                 hand_data=hand_data,
#                 screen_glow_boxes=screen_glow_detected,
#             )
#
#             # -----------------------------
#             # 8. Activity Classification
#             # -----------------------------
#             for student in student_data:
#                 sid = student["id"]
#
#                 activity = self.activity_classifier.classify(
#                     student_data=student_data,
#                     phone_detections=phone_detections,
#                     hands_data=hand_data,
#                     head_poses=head_pose_data,
#                 )
#                 # Getting student status from Module 8
#                 status_info = student_statuses.get(sid,{})
#                 status_label = status_info.get("status","Unknown")
#                 confidence = status_info.get("confidence",0.0)
#
#                 # Draw label
#                 label = "Active" if activity else "Idle"
#                 cv2.putText(frame, label, (student["x1"], student["y1"] - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#
#                 # Play alert if active with phone
#                 sid = student["id"]
#                 has_phone = sid in students_with_phone_ids
#                 now = time.time()
#                 if activity and has_phone and now - last_alert.get(sid, 0) > ALERT_COOLDOWN:
#                     # threading.Thread(target=lambda: winsound.Beep(1000,250), daemon=True).start()
#                     last_alert[sid] = now
#                     alert_queue.put(sid)
#
#             # -----------------------------
#             # Display FPS
#             # -----------------------------
#             cv2.putText(frame, f"FPS: {self.processor.fps:.1f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#             # Show frame
#             cv2.imshow("Integrated Detection + Activity Classification", frame)
#
#             # Quit on 'q'
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         self.processor.stop()
#         cv2.destroyAllWindows()
#
#
# # -------------------------------
# # ENTRY POINT
# # -------------------------------
# if __name__ == "__main__":
#     system = DetectionSystem(video_source=0)
#     system.run()