# module1_activity_classification.py

import cv2
import numpy as np
from typing import List, Dict

# =====================================
# Utility function: IoU
# =====================================
def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# =====================================
# Module 1 – Activity Classifier
# =====================================
class ActivityClassifier:

    def __init__(self, screen_glow_threshold: float = 50):
        self.screen_glow_threshold = screen_glow_threshold

    def classify(self,
                 student_data: List[Dict],
                 phone_detections: List[Dict],
                 hands_data: List[Dict],
                 head_poses: List[Dict]) -> Dict[int, str]:

        activity_dict = {}

        for student in student_data:
            s_id = student["id"]
            s_box = (student["x1"], student["y1"], student["x2"], student["y2"])
            activity = "Normal"

            # Check if any phone overlaps student
            active_phone = None
            for phone in phone_detections:
                p_box = phone["bbox"]
                overlap = iou(s_box, p_box)
                if overlap > 0.2:
                    active_phone = phone
                    break

            if active_phone:
                # Check if any hand is near phone
                hand_near_phone = False
                for hand in hands_data:
                    h_box = (hand["x1"], hand["y1"], hand["x2"], hand["y2"])
                    if iou(h_box, active_phone["bbox"]) > 0.1:
                        hand_near_phone = True
                        break

                # Screen glow check
                screen_glow = False
                if "brightness" in active_phone:
                    screen_glow = active_phone["brightness"] > self.screen_glow_threshold

                # Head pose check
                head_towards_phone = False
                for hp in head_poses:
                    if hp["label"] in ["Looking Down", "Looking Right", "Looking Left"]:
                        head_towards_phone = True

                # Activity determination
                if hand_near_phone and head_towards_phone:
                    activity = "Actively Using Phone"
                else:
                    activity = "Just Holding Phone"

            activity_dict[s_id] = activity

        return activity_dict

    def visualize(self, frame: np.ndarray, student_data: List[Dict], activity_results: Dict[int, str],
                  phone_detections: List[Dict] = []) -> np.ndarray:
        """
        Draw bounding boxes and activity labels on frame
        """
        for student in student_data:
            x1, y1, x2, y2 = student["x1"], student["y1"], student["x2"], student["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = activity_results.get(student["id"], "Unknown")
            cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for phone in phone_detections:
            x1, y1, x2, y2 = phone["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return frame


# =====================================
# Standalone test runner
# =====================================
# if __name__ == "__main__":
#     print("[INFO] Running standalone test for ActivityClassifier...")
#
#     # Dummy test data
#     student_data = [
#         {"id": 1, "x1": 100, "y1": 100, "x2": 200, "y2": 300},
#         {"id": 2, "x1": 300, "y1": 100, "x2": 400, "y2": 300}
#     ]
#     phone_detections = [
#         {"bbox": (110, 200, 150, 250), "confidence": 0.95, "brightness": 120},
#     ]
#     hands_data = [
#         {"id": 1, "x1": 105, "y1": 180, "x2": 155, "y2": 260},
#     ]
#     head_poses = [
#         {"pitch": 20, "yaw": 0, "roll": 0, "label": "Looking Down"}
#     ]
#
#     classifier = ActivityClassifier(screen_glow_threshold=50)
#     results = classifier.classify(student_data, phone_detections, hands_data, head_poses)
#     print("Classification Results:")
#     for s_id, activity in results.items():
#         print(f"Student {s_id}: {activity}")
#
#     # Optional visualization
#     frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     frame = classifier.visualize(frame, student_data, results, phone_detections)
#     cv2.imshow("Activity Classification Test", frame)
#     print("[INFO] Press 'q' to exit visualization")
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cv2.destroyAllWindows()