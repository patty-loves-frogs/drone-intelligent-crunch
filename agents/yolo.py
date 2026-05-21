from ultralytics import YOLO
import cv2
import argparse
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# ============================================
# MODELE
# ============================================

model = YOLO("weights/YOLO_V8/yolov8n-pose.pt")

# ============================================
# PARAMETRES
# ============================================

ANALYSIS_STRIDE = 5
EVENT_WINDOW_SECONDS = 2
MAX_FRAMES_PER_EVENT = 3
OUTPUT_ROOT = "outputs/runs"


# ============================================
# JSON SAFE
# ============================================

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ============================================
# OUTILS
# ============================================

def visible(kpt, threshold=0.35):
    return kpt[2] > threshold


def image_position(box, frame_shape):
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    horizontal = "gauche" if cx < frame_w / 3 else "droite" if cx > 2 * frame_w / 3 else "centre"
    vertical = "haut" if cy < frame_h / 3 else "bas" if cy > 2 * frame_h / 3 else "milieu"

    return f"{vertical}-{horizontal}"


def is_valid_person(box, kpts, frame_shape):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if h <= 0 or w <= 0:
        return False

    frame_h, frame_w = frame_shape[:2]
    area = w * h
    frame_area = frame_w * frame_h

    if area < 0.0008 * frame_area:
        return False

    ratio = w / h

    if ratio < 0.15 or ratio > 3.5:
        return False

    visible_points = sum(1 for k in kpts if k[2] > 0.35)

    if visible_points < 5:
        return False

    return True


def classify_posture(box, kpts):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if h <= 0:
        return "INCERTAIN", 0.95

    ratio = w / h
    visible_points = sum(1 for k in kpts if visible(k))

    if ratio > 1.15:
        uncertainty = max(0.05, 0.35 - (ratio - 1.15))
        return "ALLONGE", round(uncertainty, 2)

    if ratio < 1.05:
        confidence_bonus = min(visible_points / 17, 1.0)
        uncertainty = 0.45 - (confidence_bonus * 0.35)
        uncertainty = max(0.05, uncertainty)
        return "DEBOUT", round(uncertainty, 2)

    uncertainty = 0.60
    if visible_points < 6:
        uncertainty = 0.80

    return "INCERTAIN", uncertainty


def risk_level(posture, uncertainty):
    if posture == "ALLONGE" and uncertainty <= 0.40:
        return "ELEVE"
    if posture == "ALLONGE":
        return "MOYEN"
    if posture == "INCERTAIN":
        return "A_VERIFIER"
    return "FAIBLE"


def needs_vlm(posture, uncertainty):
    return posture in ["ALLONGE", "INCERTAIN"] or uncertainty > 0.40


def draw_big_label(img, label, uncertainty, x1, y1):
    text = f"{label} | incert. {uncertainty:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(x1, 5)
    y = max(y1 - 10, th + 20)

    cv2.rectangle(
        img,
        (x, y - th - 12),
        (x + tw + 20, y + 8),
        (255, 0, 0),
        -1
    )

    cv2.putText(
        img,
        text,
        (x + 10, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================
# FRAME PROCESSING
# ============================================

def process_frame(frame, frame_index=0, timestamp_sec=0.0):
    results = model(
        frame,
        conf=0.35,
        iou=0.50,
        imgsz=960,
        verbose=False
    )

    annotated = results[0].plot(labels=False)
    detections = []

    boxes = results[0].boxes.xyxy.cpu().numpy()

    if results[0].keypoints is None:
        return annotated, detections

    keypoints = results[0].keypoints.data.cpu().numpy()
    yolo_confs = results[0].boxes.conf.cpu().numpy()

    for idx, (box, kpts, yolo_conf) in enumerate(zip(boxes, keypoints, yolo_confs)):
        if not is_valid_person(box, kpts, frame.shape):
            continue

        x1, y1, x2, y2 = map(int, box)

        posture, uncertainty = classify_posture(box, kpts)

        detection = {
            "id": int(idx),
            "frame_index": int(frame_index),
            "timestamp_sec": float(timestamp_sec),
            "class": "person",
            "posture": posture,
            "uncertainty_score": float(uncertainty),
            "risk_level": risk_level(posture, uncertainty),
            "position_in_image": image_position(box, frame.shape),
            "yolo_confidence": float(yolo_conf),
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_center": [
                float((x1 + x2) / 2),
                float((y1 + y2) / 2)
            ],
            "bbox_size": {
                "width": float(x2 - x1),
                "height": float(y2 - y1)
            },
            "visible_keypoints_count": int(sum(1 for k in kpts if k[2] > 0.35)),
            "needs_vlm": bool(needs_vlm(posture, uncertainty))
        }

        detections.append(detection)

        draw_big_label(
            annotated,
            posture,
            uncertainty,
            x1,
            y1
        )

    return annotated, detections


# ============================================
# RUN FOLDERS
# ============================================

def create_run_dirs(video_path):
    video_stem = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{video_stem}"

    run_dir = Path(OUTPUT_ROOT) / run_name
    raw_dir = run_dir / "raw"
    annotated_dir = run_dir / "annotated"

    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "raw_images_dir": str(raw_dir),
        "annotated_images_dir": str(annotated_dir),
        "json_path": str(run_dir / "analysis.json")
    }


# ============================================
# VIDEO
# ============================================

def run_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Video introuvable")
        return {
            "video_path": path,
            "error": "Video introuvable",
            "frames": [],
            "detections": [],
            "summary": {}
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    run_info = create_run_dirs(path)

    frame_count = 0
    event_id = 0
    event_window_frames = int(fps * EVENT_WINDOW_SECONDS)

    current_event = []
    last_detection_frame = None

    saved_frames = []
    all_detections = []

    def save_event(event_frames, event_id):
        nonlocal saved_frames, all_detections

        if not event_frames:
            return

        def priority(item):
            detections = item["detections"]
            score = 0

            for d in detections:
                if d["posture"] == "ALLONGE":
                    score += 100
                if d["posture"] == "INCERTAIN":
                    score += 50
                if d["needs_vlm"]:
                    score += 30

                score += d["yolo_confidence"] * 10
                score += (1 - d["uncertainty_score"]) * 10

            return score

        ranked = sorted(event_frames, key=priority, reverse=True)
        selected = ranked[:MAX_FRAMES_PER_EVENT]

        for i, item in enumerate(selected):
            annotated = item["annotated"]
            raw_frame = item["raw_frame"]
            detections = item["detections"]
            frame_index = item["frame_index"]
            timestamp_sec = item["timestamp_sec"]

            base_name = f"event_{event_id:03d}_frame_{frame_index:06d}_{i}.jpg"

            raw_path = str(Path(run_info["raw_images_dir"]) / base_name)
            annotated_path = str(Path(run_info["annotated_images_dir"]) / base_name)

            cv2.imwrite(raw_path, raw_frame)
            cv2.imwrite(annotated_path, annotated)

            frame_record = {
                "event_id": int(event_id),
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp_sec),
                "raw_image_path": raw_path,
                "annotated_image_path": annotated_path,
                "detections": detections,
                "summary": {
                    "persons_detected": int(len(detections)),
                    "allonge_count": int(sum(1 for d in detections if d["posture"] == "ALLONGE")),
                    "debout_count": int(sum(1 for d in detections if d["posture"] == "DEBOUT")),
                    "incertain_count": int(sum(1 for d in detections if d["posture"] == "INCERTAIN")),
                    "needs_vlm": bool(any(bool(d["needs_vlm"]) for d in detections))
                }
            }

            saved_frames.append(frame_record)
            all_detections.extend(detections)

            #print(f"[EVENT {event_id}] raw saved: {raw_path}")
            #print(f"[EVENT {event_id}] annotated saved: {annotated_path}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % ANALYSIS_STRIDE != 0:
            continue

        timestamp_sec = round(frame_count / fps, 2)

        annotated, detections = process_frame(
            frame,
            frame_index=frame_count,
            timestamp_sec=timestamp_sec
        )

        has_detection = len(detections) > 0

        if has_detection:
            current_event.append({
                "frame_index": frame_count,
                "timestamp_sec": timestamp_sec,
                "raw_frame": frame.copy(),
                "annotated": annotated.copy(),
                "detections": detections
            })

            last_detection_frame = frame_count

        if current_event and (
            frame_count - last_detection_frame > event_window_frames
        ):
            save_event(current_event, event_id)
            event_id += 1
            current_event = []
            last_detection_frame = None

        cv2.imshow("Video posture intelligent sampling", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if current_event:
        save_event(current_event, event_id)

    cap.release()
    cv2.destroyAllWindows()

    yolo_json = {
        "run_name": run_info["run_name"],
        "video_path": path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_paths": {
            "run_dir": run_info["run_dir"],
            "raw_images_dir": run_info["raw_images_dir"],
            "annotated_images_dir": run_info["annotated_images_dir"],
            "json_path": run_info["json_path"]
        },
        "parameters": {
            "model": "yolov8n-pose.pt",
            "analysis_stride": ANALYSIS_STRIDE,
            "event_window_seconds": EVENT_WINDOW_SECONDS,
            "max_frames_per_event": MAX_FRAMES_PER_EVENT,
            "fps": float(fps)
        },
        "frames_analyzed": int(len(saved_frames)),
        "frames": saved_frames,
        "detections": all_detections,
        "summary": {
            "persons_detected": int(len(all_detections)),
            "standing_count": int(sum(1 for d in all_detections if d["posture"] == "DEBOUT")),
            "lying_count": int(sum(1 for d in all_detections if d["posture"] == "ALLONGE")),
            "uncertain_count": int(sum(1 for d in all_detections if d["posture"] == "INCERTAIN")),
            "vlm_candidate_count": int(sum(1 for d in all_detections if d["needs_vlm"])),
            "events_count": int(event_id + 1 if saved_frames else 0)
        }
    }

    yolo_json = make_json_safe(yolo_json)

    with open(run_info["json_path"], "w", encoding="utf-8") as f:
        json.dump(yolo_json, f, indent=2, ensure_ascii=False)

    print(f"Analyse terminee.")
    #print(f"Run dir: {run_info['run_dir']}")
    #print(f"JSON global: {run_info['json_path']}")

    return yolo_json


# ============================================
# NODE LANGGRAPH
# ============================================

def find_first_video(videos_dir="videos"):
    videos_path = Path(videos_dir)

    if not videos_path.exists():
        return None

    for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]:
        files = list(videos_path.glob(ext))
        if files:
            return str(files[0])

    return None


def select_vlm_candidates(yolo_json, instruction=""):
    candidates = []

    for frame in yolo_json.get("frames", []):
        detections = frame.get("detections", [])

        should_send = any(
            d.get("needs_vlm")
            or d.get("posture") == "ALLONGE"
            or d.get("posture") == "INCERTAIN"
            or d.get("uncertainty_score", 0) >= 0.45
            for d in detections
        )

        if should_send:
            candidates.append({
                "event_id": frame.get("event_id"),
                "frame_index": frame.get("frame_index"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "image_path": frame.get("raw_image_path"),
                "annotated_image_path": frame.get("annotated_image_path"),
                "detections": detections,
                "reason": "posture critique/incertaine ou incertitude élevée"
            })

    return candidates


def analyze_video_with_yolo(video_path):
    return run_video(video_path)


def yolo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    instruction = state.get("instruction", "")
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
                "vlm_candidate_count": 0
            },
            "error": "Aucune vidéo trouvée. Place une vidéo dans le dossier videos/."
        }

        return {
            **state,
            "video_path": "",
            "yolo_json": yolo_json,
            "images": [],
            "detections": [],
            "vlm_candidates": []
        }

    yolo_json = analyze_video_with_yolo(video_path)

    frames = yolo_json.get("frames", [])

    images = [
        frame.get("raw_image_path")
        for frame in frames
        if frame.get("raw_image_path")
    ]

    detections = yolo_json.get("detections", [])

    vlm_candidates = select_vlm_candidates(
        yolo_json,
        instruction
    )

    yolo_json["summary"]["vlm_candidate_count"] = len(vlm_candidates)

    return {
        **state,
        "video_path": video_path,
        "yolo_json": yolo_json,
        "images": images,
        "detections": detections,
        "vlm_candidates": vlm_candidates
    }


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Chemin vers la vidéo à analyser"
    )

    args = parser.parse_args()

    run_video(args.video)