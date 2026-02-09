import math

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxA_area + boxB_area - interArea)


# ------------------------------
# CENTER DISTANCE
# ------------------------------
def center_distance(boxA, boxB):
    ax = (boxA[0] + boxA[2]) / 2
    ay = (boxA[1] + boxA[3]) / 2

    bx = (boxB[0] + boxB[2]) / 2
    by = (boxB[1] + boxB[3]) / 2

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ========================================================================
# MAIN ASSOCIATION FUNCTION
# ========================================================================
def associate_phones_to_students(
    student_boxes,
    phone_boxes,
    hand_boxes=None,
    head_pose=None,
    screen_glow=None,
):

    if hand_boxes is None:
        hand_boxes = []
    if head_pose is None:
        head_pose = {}
    if screen_glow is None:
        screen_glow = []

    students_with_phone = set()
    associations = []

    # -----------------------------------------------
    # ASSOCIATION PIPELINE
    # -----------------------------------------------
    for phone in phone_boxes:
        phone_box = phone["bbox"]

        best_student = None
        best_score = 0.0
        best_details = None

        for stu in student_boxes:

            stu_box = (stu["x1"], stu["y1"], stu["x2"], stu["y2"])
            stu_id = stu["id"]

            # STEP 1 — IoU score
            iou_score = compute_iou(phone_box, stu_box)

            # STEP 2 — Distance score (closer → higher)
            dist = center_distance(phone_box, stu_box)
            distance_score = 1 / (1 + dist)  # normalize

            # INITIAL BASE SCORE
            score = (0.7 * iou_score) + (0.3 * distance_score)

            # ------------------------------------------------------
            # OPTIONAL BOOSTS:
            # ------------------------------------------------------

            # A) If phone intersects a hand linked to same student
            hand_bonus = False
            for hand in hand_boxes:
                if hand["student_id"] == stu_id:
                    if compute_iou(phone_box, hand["bbox"]) > 0.02:
                        score += 0.15
                        hand_bonus = True

            # B) Screen glow near student face
            glow_bonus = False
            for glow in screen_glow:
                if compute_iou(phone_box, glow) > 0.03:
                    score += 0.10
                    glow_bonus = True

            # C) If student head pose is strongly downward (phone looking)
            if stu_id in head_pose:
                yaw = head_pose[stu_id]["yaw"]
                pitch = head_pose[stu_id]["pitch"]

                if pitch > 15:   # looking down
                    score += 0.10

            # ------------------------------------------------------
            # PICK BEST STUDENT
            # ------------------------------------------------------
            if score > best_score:
                best_score = score
                best_student = stu_id
                best_details = {
                    "student_id": stu_id,
                    "phone_box": phone_box,
                    "score": float(best_score),
                    "iou": float(iou_score),
                    "distance_score": float(distance_score),
                    "glow_bonus": glow_bonus,
                    "hand_bonus": hand_bonus
                }

        # Threshold so random detections do not associate wrongly
        if best_student is not None and best_score >= 0.20:
            students_with_phone.add(best_student)
            associations.append(best_details)

    return students_with_phone, associations