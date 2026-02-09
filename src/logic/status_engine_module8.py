import time
import math

class StudentStatusLogicEngine:

    def __init__(
        self,
        phone_confirm_time=1.5,
        suspicious_confirm_time=2.0,
        inactive_timeout=3.0
    ):
        self.phone_confirm_time = phone_confirm_time
        self.suspicious_confirm_time = suspicious_confirm_time
        self.inactive_timeout = inactive_timeout

        # Internal memory per student
        self.student_memory = {}

    # --------------------------------------------------
    # MAIN UPDATE FUNCTION (call once per frame)
    # --------------------------------------------------
    def update(
        self,
        students,
        students_with_phone,
        head_pose_data=None,
        hand_data=None,
        screen_glow_boxes=None
    ):

        now = time.time()

        head_pose_data = head_pose_data or {}
        hand_data = hand_data or []
        screen_glow_boxes = screen_glow_boxes or []

        current_ids = {s["id"] for s in students}

        # ----------------------------------------------
        # HANDLE INACTIVE STUDENTS
        # ----------------------------------------------
        for sid in list(self.student_memory.keys()):
            if sid not in current_ids:
                if now - self.student_memory[sid]["last_seen"] > self.inactive_timeout:
                    self.student_memory[sid]["status"] = "INACTIVE"
                    self.student_memory[sid]["confidence"] = 0.0

        # ----------------------------------------------
        # PROCESS EACH STUDENT
        # ----------------------------------------------
        for student in students:
            sid = student["id"]

            if sid not in self.student_memory:
                self.student_memory[sid] = {
                    "status": "ATTENTIVE",
                    "confidence": 0.5,
                    "since": now,
                    "last_seen": now,
                    "phone_timer": None,
                    "suspicious_timer": None
                }

            mem = self.student_memory[sid]
            mem["last_seen"] = now

            # ==========================================
            # RULE 1 — USING PHONE (highest priority)
            # ==========================================
            if sid in students_with_phone:
                if mem["phone_timer"] is None:
                    mem["phone_timer"] = now

                if now - mem["phone_timer"] >= self.phone_confirm_time:
                    mem["status"] = "USING_PHONE"
                    mem["confidence"] = self._calculate_phone_confidence(
                        sid, head_pose_data, screen_glow_boxes
                    )
                    continue
            else:
                mem["phone_timer"] = None

            # ==========================================
            # RULE 2 — SUSPICIOUS
            # ==========================================
            suspicious = False

            if sid in head_pose_data:
                pitch = head_pose_data[sid]["pitch"]
                if pitch > 15:  # looking down
                    suspicious = True

            if suspicious:
                if mem["suspicious_timer"] is None:
                    mem["suspicious_timer"] = now

                if now - mem["suspicious_timer"] >= self.suspicious_confirm_time:
                    mem["status"] = "SUSPICIOUS"
                    mem["confidence"] = 0.6
                    continue
            else:
                mem["suspicious_timer"] = None

            # ==========================================
            # RULE 3 — ATTENTIVE (default)
            # ==========================================
            mem["status"] = "ATTENTIVE"
            mem["confidence"] = 0.7

        # ----------------------------------------------
        # BUILD OUTPUT
        # ----------------------------------------------
        output = {}
        for sid, mem in self.student_memory.items():
            output[sid] = {
                "status": mem["status"],
                "confidence": round(mem["confidence"], 2),
                "since": mem["since"]
            }

        return output

    # --------------------------------------------------
    # CONFIDENCE CALCULATION
    # --------------------------------------------------
    def _calculate_phone_confidence(self, sid, head_pose_data, screen_glow_boxes):
        confidence = 0.7

        if sid in head_pose_data:
            if head_pose_data[sid]["pitch"] > 15:
                confidence += 0.1

        if screen_glow_boxes:
            confidence += 0.1

        return min(confidence, 1.0)