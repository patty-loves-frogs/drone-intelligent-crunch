import os
from typing import Dict, Any, List

import cv2
from ultralytics import YOLO


# ============================================
# MODELE YOLO POSE
# ============================================

try:
    _model = YOLO("yolov8n-pose.pt")
except Exception as e:
    _model = None
    print(f"[yolo_pose] Erreur chargement YOLO Pose: {e}")


PERSON_KEYWORDS = [
    "personne", "quelqu'un", "quelqu un",
    "homme", "femme", "gens",
    "people", "person", "individual", "human",
    "blessé", "blesse", "victime",
    "allongé", "allonge", "debout", "posture",
]


# ============================================
# OUTILS POSTURE
# ============================================

def visible(kpt, threshold=0.35) -> bool:
    return kpt[2] > threshold


def is_valid_person(box, kpts, frame_shape) -> bool:
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    frame_h, frame_w = frame_shape[:2]
    area = w * h
    frame_area = frame_w * frame_h

    # Trop petit = faux positif
    if area < 0.0008 * frame_area:
        return False

    if h <= 0 or w <= 0:
        return False

    ratio = w / h

    # bbox absurde
    if ratio < 0.15 or ratio > 3.5:
        return False

    # Pas assez de keypoints visibles
    visible_points = sum(1 for k in kpts if k[2] > 0.35)

    if visible_points < 5:
        return False

    return True


def classify_posture(box, kpts) -> str:
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    if h <= 0:
        return "INCERTAIN"

    ratio = w / h

    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_SHOULDER, R_SHOULDER = 5, 6

    hips_visible = visible(kpts[L_HIP]) or visible(kpts[R_HIP])
    knees_visible = visible(kpts[L_KNEE]) or visible(kpts[R_KNEE])
    shoulders_visible = visible(kpts[L_SHOULDER]) or visible(kpts[R_SHOULDER])

    # Personne plutôt horizontale
    if ratio > 1.20:
        return "ALLONGE"

    # Personne plutôt verticale
    if ratio < 0.75:
        return "DEBOUT"

    if ratio < 0.95 and shoulders_visible and (hips_visible or knees_visible):
        return "DEBOUT"

    return "INCERTAIN"


# ============================================
# SCAN DES IMAGES
# ============================================

def _scan_frames() -> List[str]:
    frames_dir = "frames"
    extensions = (".jpg", ".jpeg", ".png", ".webp")

    images: List[str] = []

    if os.path.isdir(frames_dir):
        for filename in sorted(os.listdir(frames_dir)):
            if filename.lower().endswith(extensions):
                images.append(os.path.join(frames_dir, filename))

    return images


# ============================================
# DETECTION POSTURE SUR UNE IMAGE
# ============================================

def detect_postures(image_path: str) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    if _model is None:
        return [{
            "image": image_path,
            "label": "error",
            "confidence": 0.0,
            "bounding_box": [],
            "error": "YOLO Pose model not loaded",
        }]

    img = cv2.imread(image_path)

    if img is None:
        return [{
            "image": image_path,
            "label": "error",
            "confidence": 0.0,
            "bounding_box": [],
            "error": "Image introuvable ou illisible",
        }]

    try:
        results = _model(
            img,
            conf=0.35,
            iou=0.50,
            imgsz=960,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()

        for box, conf, kpts in zip(boxes, confs, keypoints):
            if not is_valid_person(box, kpts, img.shape):
                continue

            posture = classify_posture(box, kpts)

            detections.append({
                "image": image_path,
                "label": "person",
                "posture": posture,
                "confidence": round(float(conf), 3),
                "bounding_box": [round(float(x), 2) for x in box],
            })

    except Exception as e:
        detections.append({
            "image": image_path,
            "label": "error",
            "confidence": 0.0,
            "bounding_box": [],
            "error": str(e),
        })

    return detections


# ============================================
# NOEUD LANGGRAPH
# ============================================

def yolo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nœud YOLO Pose :
      1. Scanne les images dans frames/
      2. Détecte les personnes avec YOLO Pose
      3. Classe leur posture : DEBOUT, ALLONGE ou INCERTAIN
      4. Envoie au VLM les images incertaines ou sans détection fiable
    """

    images = _scan_frames()

    instruction: str = state.get("instruction", "")
    detections: List[Dict[str, Any]] = []
    vlm_candidates: List[str] = []

    if not images:
        return {
            "images": images,
            "detections": detections,
            "vlm_candidates": vlm_candidates,
        }

    asks_about_person = any(
        kw in instruction.lower()
        for kw in PERSON_KEYWORDS
    )

    for img_path in images:
        img_detections = detect_postures(img_path)
        detections.extend(img_detections)

        max_person_conf = 0.0
        has_uncertain_posture = False
        has_person = False

        for det in img_detections:
            if det.get("label") == "person":
                has_person = True
                max_person_conf = max(
                    max_person_conf,
                    det.get("confidence", 0.0)
                )

                if det.get("posture") == "INCERTAIN":
                    has_uncertain_posture = True

        # Fallback VLM si la mission concerne une personne
        if asks_about_person:
            if not has_person:
                vlm_candidates.append(img_path)
            elif max_person_conf < 0.60:
                vlm_candidates.append(img_path)
            elif has_uncertain_posture:
                vlm_candidates.append(img_path)

    return {
        "images": images,
        "detections": detections,
        "vlm_candidates": vlm_candidates,
    }