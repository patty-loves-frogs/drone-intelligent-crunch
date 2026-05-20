import os
import json
from typing import Dict, Any, List, Optional

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = "videos"
OUTPUT_DIR = "outputs"
RAW_FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
ANNOTATED_FRAMES_DIR = os.path.join(OUTPUT_DIR, "annotated_frames")
YOLO_JSON_PATH = os.path.join(OUTPUT_DIR, "yolo_pose_results.json")

# Analyse 1 frame toutes les N frames.
# Augmente si la vidéo est longue, diminue si tu veux plus de précision.
FRAME_STEP = 10

MODEL_PATH = "yolov8n-pose.pt"


# ============================================================
# CHARGEMENT MODELE
# ============================================================

try:
    _model = YOLO(MODEL_PATH)
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


# ============================================================
# OUTILS POSTURE
# ============================================================

def visible(kpt, threshold: float = 0.35) -> bool:
    """Retourne True si un keypoint YOLO Pose est suffisamment visible."""
    return float(kpt[2]) > threshold


def is_valid_person(box, kpts, frame_shape) -> bool:
    """
    Filtre les faux positifs :
    - bbox trop petite
    - ratio largeur/hauteur absurde
    - pas assez de keypoints visibles
    """
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    frame_h, frame_w = frame_shape[:2]
    area = w * h
    frame_area = frame_w * frame_h

    if area < 0.0008 * frame_area:
        return False

    if h <= 0 or w <= 0:
        return False

    ratio = w / h

    if ratio < 0.15 or ratio > 3.5:
        return False

    visible_points = sum(1 for k in kpts if float(k[2]) > 0.35)

    if visible_points < 5:
        return False

    return True


def classify_posture(box, kpts) -> str:
    """
    Classe une personne détectée en :
    - ALLONGE
    - DEBOUT
    - INCERTAIN
    """
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

    if ratio > 1.20:
        return "ALLONGE"

    if ratio < 0.75:
        return "DEBOUT"

    if ratio < 0.95 and shoulders_visible and (hips_visible or knees_visible):
        return "DEBOUT"

    return "INCERTAIN"


def draw_posture_label(img, label: str, x1: int, y1: int) -> None:
    """Dessine le label de posture sur l'image annotée."""
    text = f"ETAT : {label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(x1, 5)
    y = max(y1 - 10, th + 20)

    cv2.rectangle(
        img,
        (x, y - th - 12),
        (x + tw + 20, y + 8),
        (255, 0, 0),
        -1,
    )

    cv2.putText(
        img,
        text,
        (x + 10, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# VIDEO INPUT
# ============================================================

def find_first_video(video_dir: str = VIDEO_DIR) -> Optional[str]:
    """Cherche automatiquement une vidéo dans le dossier videos/."""
    extensions = (".mp4", ".mov", ".avi", ".mkv")

    if not os.path.isdir(video_dir):
        return None

    for filename in sorted(os.listdir(video_dir)):
        if filename.lower().endswith(extensions):
            return os.path.join(video_dir, filename)

    return None


# ============================================================
# YOLO : ENTREE VIDEO_PATH -> SORTIE JSON
# ============================================================

def analyze_video_with_yolo(video_path: str) -> Dict[str, Any]:
    """
    Entrée :
        video_path : chemin vers une vidéo

    Sortie :
        JSON Python contenant :
        - video_path
        - frames analysées
        - path vers frame brute
        - path vers frame annotée
        - score de confiance
        - bbox
        - posture
        - résumé global
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_FRAMES_DIR, exist_ok=True)
    os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)

    yolo_json: Dict[str, Any] = {
        "video_path": video_path,
        "frame_step": FRAME_STEP,
        "frames_analyzed": 0,
        "frames": [],
        "detections": [],
        "summary": {
            "persons_detected": 0,
            "standing_count": 0,
            "lying_count": 0,
            "uncertain_count": 0,
            "low_confidence_count": 0,
        },
        "error": None,
    }

    if _model is None:
        yolo_json["error"] = "YOLO Pose model not loaded"
        return yolo_json

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        yolo_json["error"] = f"Video introuvable ou illisible: {video_path}"
        return yolo_json

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_STEP != 0:
            frame_id += 1
            continue

        raw_frame_path = os.path.join(RAW_FRAMES_DIR, f"frame_{saved_id:04d}.jpg")
        annotated_frame_path = os.path.join(
            ANNOTATED_FRAMES_DIR,
            f"frame_{saved_id:04d}.jpg",
        )

        frame_json: Dict[str, Any] = {
            "frame_id": frame_id,
            "frame_index": saved_id,
            "image_path": raw_frame_path,
            "annotated_image_path": annotated_frame_path,
            "detections": [],
        }

        try:
            cv2.imwrite(raw_frame_path, frame)

            results = _model(
                frame,
                conf=0.35,
                iou=0.50,
                imgsz=960,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot(labels=False)

            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()

                for person_id, (box, conf, kpts) in enumerate(zip(boxes, confs, keypoints)):
                    if not is_valid_person(box, kpts, frame.shape):
                        continue

                    posture = classify_posture(box, kpts)
                    confidence = round(float(conf), 3)
                    bbox = [round(float(x), 2) for x in box]

                    x1, y1, x2, y2 = map(int, box)
                    draw_posture_label(annotated, posture, x1, y1)

                    detection = {
                        "video_path": video_path,
                        "frame_id": frame_id,
                        "frame_index": saved_id,
                        "image_path": raw_frame_path,
                        "annotated_image_path": annotated_frame_path,
                        "person_id": person_id,
                        "label": "person",
                        "posture": posture,
                        "confidence": confidence,
                        "bounding_box": bbox,
                    }

                    frame_json["detections"].append(detection)
                    yolo_json["detections"].append(detection)

                    yolo_json["summary"]["persons_detected"] += 1

                    if posture == "DEBOUT":
                        yolo_json["summary"]["standing_count"] += 1
                    elif posture == "ALLONGE":
                        yolo_json["summary"]["lying_count"] += 1
                    else:
                        yolo_json["summary"]["uncertain_count"] += 1

                    if confidence < 0.60:
                        yolo_json["summary"]["low_confidence_count"] += 1

            cv2.imwrite(annotated_frame_path, annotated)

        except Exception as e:
            frame_json["error"] = str(e)

        yolo_json["frames"].append(frame_json)
        yolo_json["frames_analyzed"] += 1

        saved_id += 1
        frame_id += 1

    cap.release()

    with open(YOLO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(yolo_json, f, indent=2, ensure_ascii=False)

    return yolo_json


def select_vlm_candidates(yolo_json: Dict[str, Any], instruction: str) -> List[str]:
    """
    Sélectionne les images à envoyer au VLM.
    Règles :
    - uniquement si la demande concerne une personne
    - posture INCERTAIN
    - confiance faible
    - frame sans détection alors que la mission parle de personne
    """
    asks_about_person = any(
        kw in instruction.lower()
        for kw in PERSON_KEYWORDS
    )

    if not asks_about_person:
        return []

    candidates: List[str] = []

    for frame in yolo_json.get("frames", []):
        detections = frame.get("detections", [])

        if not detections:
            candidates.append(frame["image_path"])
            continue

        for det in detections:
            if det.get("label") != "person":
                continue

            if det.get("confidence", 0.0) < 0.60:
                candidates.append(det["image_path"])

            if det.get("posture") == "INCERTAIN":
                candidates.append(det["image_path"])

    # supprime les doublons en gardant l'ordre
    return list(dict.fromkeys(candidates))


# ============================================================
# NOEUD LANGGRAPH
# ============================================================

def yolo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent YOLO.

    Contrat :
        Entrée :
            state["video_path"] ou première vidéo trouvée dans videos/

        Sortie :
            {
                "yolo_json": JSON structuré,
                "images": paths des frames extraites,
                "detections": liste globale des détections,
                "vlm_candidates": paths des images à envoyer au VLM
            }
    """
    instruction: str = state.get("instruction", "")

    video_path = state.get("video_path") or find_first_video()

    if not video_path:
        yolo_json = {
            "video_path": "",
            "frames_analyzed": 0,
            "frames": [],
            "detections": [],
            "summary": {
                "persons_detected": 0,
                "standing_count": 0,
                "lying_count": 0,
                "uncertain_count": 0,
                "low_confidence_count": 0,
            },
            "error": "Aucune vidéo trouvée. Place une vidéo dans le dossier videos/.",
        }

        return {
            "video_path": "",
            "yolo_json": yolo_json,
            "images": [],
            "detections": [],
            "vlm_candidates": [],
        }

    yolo_json = analyze_video_with_yolo(video_path)

    images = [
        frame["image_path"]
        for frame in yolo_json.get("frames", [])
        if frame.get("image_path")
    ]

    detections = yolo_json.get("detections", [])
    vlm_candidates = select_vlm_candidates(yolo_json, instruction)

    return {
        "video_path": video_path,
        "yolo_json": yolo_json,
        "images": images,
        "detections": detections,
        "vlm_candidates": vlm_candidates,
    }
