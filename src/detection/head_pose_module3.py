# Integrated Version

import time
import cv2
import numpy as np
import mediapipe as mp
from math import atan2, degrees

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing_module9 import VideoProcessor  # Module 9

class HeadPoseDetector:
    def __init__(self, max_faces=4, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )

        self.model_points_3D = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float32)

    def process_frame(self, frame):
        if frame is None:
            return None, []

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        head_poses = []

        if results.multi_face_landmarks:
            for faceLM in results.multi_face_landmarks:
                pts_2D = np.array([
                    [faceLM.landmark[1].x * w, faceLM.landmark[1].y * h],     # Nose
                    [faceLM.landmark[152].x * w, faceLM.landmark[152].y * h], # Chin
                    [faceLM.landmark[33].x * w, faceLM.landmark[33].y * h],   # Left eye
                    [faceLM.landmark[263].x * w, faceLM.landmark[263].y * h], # Right eye
                    [faceLM.landmark[78].x * w, faceLM.landmark[78].y * h],   # Left mouth
                    [faceLM.landmark[308].x * w, faceLM.landmark[308].y * h]  # Right mouth
                ], dtype=np.float32)

                focal_length = w
                camera_matrix = np.array(
                    [[focal_length, 0, w/2],
                     [0, focal_length, h/2],
                     [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.zeros((4,1))

                success, rvec, tvec = cv2.solvePnP(self.model_points_3D, pts_2D, camera_matrix, dist_coeffs)

                if not success:
                    continue

                rmat, _ = cv2.Rodrigues(rvec)
                sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)

                if sy < 1e-6:
                    pitch = atan2(-rmat[1,2], rmat[1,1])
                    yaw = atan2(-rmat[2,0], sy)
                    roll = 0.0
                else:
                    pitch = atan2(rmat[2,1], rmat[2,2])
                    yaw = atan2(-rmat[2,0], sy)
                    roll = atan2(rmat[1,0], rmat[0,0])

                pitch = degrees(pitch); yaw = degrees(yaw); roll = degrees(roll)
                label = self.get_orientation_label(pitch, yaw)
                head_poses.append({"pitch": pitch, "yaw": yaw, "roll": roll, "label": label})

                x_min = int(min([lm.x for lm in faceLM.landmark]) * w)
                y_min = int(min([lm.y for lm in faceLM.landmark]) * h)
                x_max = int(max([lm.x for lm in faceLM.landmark]) * w)
                y_max = int(max([lm.y for lm in faceLM.landmark]) * h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
                cv2.putText(frame, label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        return frame, head_poses

    @staticmethod
    def get_orientation_label(pitch, yaw):
        if pitch > 15:
            return "Looking Up"
        elif pitch < -15:
            return "Looking Down"
        elif yaw > 15:
            return "Looking Right"
        elif yaw < -15:
            return "Looking Left"
        else:
            return "Forward"


def run_head_pose_detection(camera_index=0, use_threaded=True, width=640, height=480, wait_timeout=5.0):

    print("[INFO] Initializing VideoProcessor (Module 9)...")
    # Try threaded first
    processor = VideoProcessor(width=width, height=height, threaded=use_threaded, output_format="bgr")
    try:
        processor.capture_video(camera_index)
    except Exception as e:
        print("[WARN] VideoProcessor.capture_video() failed with exception:", e)
        if use_threaded:
            print("[INFO] Trying again with threaded=False fallback.")
            processor = VideoProcessor(width=width, height=height, threaded=False, output_format="bgr")
            try:
                processor.capture_video(camera_index)
            except Exception as e2:
                print("[ERROR] Failed to open camera even with non-threaded capture:", e2)
                print("Make sure camera is connected, not used by other apps, and try different camera_index values (0,1,...).")
                return

    detector = HeadPoseDetector()
    start_wait = time.time()

    print("[INFO] Waiting for first frames... (timeout {:.1f}s)".format(wait_timeout))
    frame = None
    while True:
        ret, frame = processor.read_frame()
        if ret and frame is not None and getattr(frame, "shape", None):
            break
        if time.time() - start_wait > wait_timeout:
            print("[ERROR] No frames received from camera after {:.1f}s.".format(wait_timeout))
            print("Possible causes: camera in use, permissions blocked, wrong camera_index.")
            # Clean up and exit
            try:
                processor.stop()
            except:
                pass
            return
        time.sleep(0.05)  # small delay while waiting

    print("[INFO] Camera started successfully — entering main loop. Press 'q' to quit.")
    last_print = time.time()
    try:
        while True:
            ret, frame = processor.read_frame()
            if not ret or frame is None:
                # Skip until next available frame
                time.sleep(0.001)
                continue

            # Preprocess is available from VideoProcessor; it resizes and returns BGR/RGB depending on config
            processed_frame = processor.preprocess(frame)

            output_frame, poses = detector.process_frame(processed_frame)

            # Draw FPS (processor.fps updated by the thread)
            fps = getattr(processor, "fps", 0.0)
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Head Pose Detection (MediaPipe) - press 'q' to exit", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Optional: print a short status every 5s
            if time.time() - last_print > 5.0:
                print(f"[INFO] Running... detected faces in last frame: {len(poses)}")
                last_print = time.time()

    finally:
        print("[INFO] Shutting down.")
        try:
            processor.stop()
        except:
            pass
        cv2.destroyAllWindows()


# if __name__ == "__main__":
#     # Try different camera_index like 0 or 1 if webcam doesn't open
#     run_head_pose_detection(camera_index=0, use_threaded=True, width=640, height=480, wait_timeout=6.0)