import os
from typing import Dict, Any, List

# Chargement global de YOLOv8
try:
    from ultralytics import YOLO
    _model = YOLO("yolov8n.pt")
except Exception as e:
    _model = None
    print(f"[yolo] Erreur chargement YOLO: {e}")

TARGET_CLASSES = {
    "person", "car", "truck", "dog",
    "backpack", "bicycle", "motorcycle",
}

PERSON_KEYWORDS = [
    "personne", "quelqu'un", "quelqu un",
    "homme", "femme", "gens",
    "people", "person", "individual", "human",
]


def _scan_frames() -> List[str]:
    """Retourne les chemins des images dans frames/."""
    frames_dir = "frames"
    extensions = (".jpg", ".jpeg", ".png")
    images: List[str] = []
    if os.path.isdir(frames_dir):
        for filename in sorted(os.listdir(frames_dir)):
            if filename.lower().endswith(extensions):
                images.append(os.path.join(frames_dir, filename))
    return images


def yolo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nœud YOLO :
      1. Scanne les images dans frames/.
      2. Détecte les objets cibles.
      3. Pour chaque image, détermine si un fallback VLM est nécessaire.
    Retourne dans le state :
      - images : liste des chemins
      - detections : résultats YOLO
      - vlm_candidates : liste des images nécessitant VLM
    """
    images = _scan_frames()
    instruction: str = state.get("instruction", "")
    detections: List[Dict[str, Any]] = []
    vlm_candidates: List[str] = []

    if not images:
        return {"images": images, "detections": detections, "vlm_candidates": vlm_candidates}

    asks_about_person = any(kw in instruction.lower() for kw in PERSON_KEYWORDS)

    for img_path in images:
        img_detections: List[Dict[str, Any]] = []
        max_person_conf = 0.0

        if _model is not None:
            try:
                results = _model(img_path)
                result = results[0]
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = result.names[cls_id]
                        if name not in TARGET_CLASSES:
                            continue
                        bbox = box.xyxy[0].tolist()
                        det = {
                            "image": img_path,
                            "label": name,
                            "confidence": round(conf, 3),
                            "bounding_box": [round(x, 2) for x in bbox],
                        }
                        img_detections.append(det)
                        if name == "person":
                            max_person_conf = max(max_person_conf, conf)
            except Exception as e:
                img_detections.append({
                    "image": img_path,
                    "label": "error",
                    "confidence": 0.0,
                    "bounding_box": [],
                    "error": str(e),
                })

        detections.extend(img_detections)

        # --- Règles VLM fallback ---
        if asks_about_person:
            if max_person_conf >= 0.60:
                pass  # Fiable, pas de VLM
            elif max_person_conf >= 0.30:
                vlm_candidates.append(img_path)  # Incertain
            else:
                vlm_candidates.append(img_path)  # Absent ou trop faible

    return {
        "images": images,
        "detections": detections,
        "vlm_candidates": vlm_candidates,
    }