# # Integrated Version
#
# import cv2
# import traceback
# import threading
# import winsound   # For Windows beep, replace for other OS
# from ultralytics import YOLO
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.preprocessing_module9 import VideoProcessor  # Module 9
#
#
# class PhoneDetector:
#     """
#     MODULE 4 – MOBILE PHONE DETECTION (Integrated with Module 9)
#     Detect phones in real-time video and give audio alerts.
#     """
#
#     def __init__(self, model_path="yolov8s.pt", confidence_threshold=0.45):
#         self.model_path = model_path
#         self.confidence_threshold = confidence_threshold
#         self.phone_class_id = 67  # COCO "cell phone"
#         self.model = None
#
#         try:
#             print("[Module 4] Loading YOLO model...")
#             self.model = YOLO(self.model_path)
#             print("[Module 4] YOLO model loaded successfully!")
#         except Exception as e:
#             print("[ERROR] Failed to load YOLO model:")
#             print(str(e))
#             traceback.print_exc()
#             self.model = None
#
#     # ------------------------
#     # Main detection method
#     # ------------------------
#     def detect_phones(self, frame):
#         if self.model is None or frame is None:
#             return []
#
#         try:
#             results = self.model.predict(frame, verbose=False)
#             detections = []
#
#             for result in results:
#                 if not hasattr(result, "boxes") or result.boxes is None:
#                     continue
#                 for box in result.boxes:
#                     cls = int(box.cls[0])
#                     conf = float(box.conf[0])
#                     if cls == self.phone_class_id and conf >= self.confidence_threshold:
#                         x1, y1, x2, y2 = box.xyxy[0].tolist()
#                         detections.append({
#                             "bbox": (int(x1), int(y1), int(x2), int(y2)),
#                             "confidence": round(conf, 3)
#                         })
#
#                         # Audio alert in a separate thread (non-blocking)
#                         threading.Thread(target=self._alert_sound, daemon=True).start()
#
#             return detections
#         except Exception as e:
#             print("[Module 4] ERROR during phone detection:", str(e))
#             traceback.print_exc()
#             return []
#
#     # ------------------------
#     # Audio alert
#     # ------------------------
#     def _alert_sound(self):
#         # 1000 Hz beep for 300 ms
#         winsound.Beep(1000, 300)
#
#     # ------------------------
#     # Drawing utility
#     # ------------------------
#     def draw_detections(self, frame, detections):
#         if frame is None:
#             return None
#         try:
#             for det in detections:
#                 x1, y1, x2, y2 = det["bbox"]
#                 conf = det["confidence"]
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"Phone {conf}", (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             return frame
#         except Exception as e:
#             print("[Module 4] ERROR while drawing boxes:", str(e))
#             traceback.print_exc()
#             return frame
#
#
# # ------------------------
# # Integrated runner with Module 9
# # ------------------------
# def run_phone_detection_with_preprocessing(use_webcam=True, video_path="students_video.mp4"):
#     # Module 9 video processor
#     processor = VideoProcessor(threaded=True, output_format="rgb")  # RGB for YOLO
#     if use_webcam:
#         processor.capture_video(0)
#     else:
#         processor.capture_video(video_path)
#
#     # Phone detector
#     detector = PhoneDetector(model_path="yolov8s.pt")
#
#     while True:
#         ret, frame = processor.read_frame()
#         if not ret or frame is None:
#             continue
#
#         # Preprocess frame (resize + RGB)
#         processed_frame = processor.preprocess(frame)
#
#         # Detect phones
#         detections = detector.detect_phones(processed_frame)
#         output_frame = detector.draw_detections(processed_frame, detections)
#
#         # Show FPS on frame
#         cv2.putText(output_frame, f"FPS: {processor.fps:.1f}", (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Display
#         cv2.imshow("Phone Detection with Audio Alerts", output_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     processor.stop()
#     cv2.destroyAllWindows()
#
#
# # ------------------------
# # Run
# # ------------------------
# if __name__ == "__main__":
#     run_phone_detection_with_preprocessing(use_webcam=True)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Upgraded Version

import cv2
import threading
import winsound
from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing_module9 import VideoProcessor


class PhoneDetector:
    def __init__(self, model_path="yolov8s.pt", confidence=0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.phone_id = 67  # COCO cell phone
        self.lock = threading.Lock()

    def detect(self, frame):
        if frame is None:
            return []

        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == self.phone_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": round(conf, 2)
                    })
                    # Non-blocking beep
                    threading.Thread(target=self._alert, daemon=True).start()

        return detections

    def _alert(self):
        try:
            winsound.Beep(1000, 150)
        except:
            pass

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Phone {conf}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame


def main():
    processor = VideoProcessor(threaded=True, output_format="bgr")
    processor.capture_video(0)  # Change 0 to video path if needed

    detector = PhoneDetector(model_path="yolov8s.pt")

    print("Starting Phone Detection... Press 'q' to quit.")
    while True:
        ret, frame = processor.read_frame()
        if not ret or frame is None:
            continue

        processed = processor.preprocess(frame)  # BGR output
        detections = detector.detect(processed)
        output = detector.draw(processed.copy(), detections)

        cv2.putText(output, f"FPS: {processor.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Phone Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.stop()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX