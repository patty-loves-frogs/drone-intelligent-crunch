import base64
from typing import Dict, Any, List
from pathlib import Path
import cv2
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

def _encode_image(image_path: str) -> str:
    """Encode une image en base64 pour l'envoyer au VLM."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def crop_face_from_image(image_path: str, bbox_xyxy: List[float]) -> tuple[np.ndarray | None, str]:
    """
    Extrait une région de visage à partir d'une image et une boîte de délimitation.
    
    Args:
        image_path: Chemin de l'image
        bbox_xyxy: [x1, y1, x2, y2] coordonnées de la boîte
    
    Returns:
        (cropped_image, output_path) ou (None, error_message)
    """
    try:
        if not Path(image_path).exists():
            return None, f"Image non trouvée: {image_path}"
        
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Impossible de lire l'image: {image_path}"
        
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        
        # Ajouter une marge pour mieux capturer le visage (30% des dimensions)
        h = y2 - y1
        w = x2 - x1
        margin_y = int(h * 0.15)
        margin_x = int(w * 0.15)
        
        # Appliquer les marges avec contrôle de limites
        y1 = max(0, y1 - margin_y)
        x1 = max(0, x1 - margin_x)
        y2 = min(img.shape[0], y2 + margin_y)
        x2 = min(img.shape[1], x2 + margin_x)
        
        # Cropper
        cropped = img[y1:y2, x1:x2].copy()
        
        if cropped.size == 0:
            return None, "Boîte de délimitation invalide"
        
        # Sauvegarder temporairement
        temp_dir = Path("temp_faces")
        temp_dir.mkdir(exist_ok=True)
        
        import time
        face_filename = f"face_{int(time.time() * 1000)}.jpg"
        face_path = temp_dir / face_filename
        
        cv2.imwrite(str(face_path), cropped)
        
        return cropped, str(face_path)
    
    except Exception as e:
        return None, f"Erreur lors du cropping: {str(e)}"


def compare_faces(face1_path: str, face2_path: str, ollama_base_url: str = None) -> Dict[str, Any]:
    """
    Compare deux visages avec un VLM pour déterminer si c'est la même personne.
    
    Args:
        face1_path: Chemin du premier visage
        face2_path: Chemin du deuxième visage
        ollama_base_url: URL de base Ollama
    
    Returns:
        {
            "same_person": bool,
            "confidence": "high" | "medium" | "low",
            "reason": str,
            "details": str
        }
    """
    try:
        if not ollama_base_url:
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Utiliser un modèle capable de traiter plusieurs images
        llm = ChatOllama(
            model="llava",
            temperature=0.2,
            base_url=ollama_base_url,
            num_ctx=2048
        )
        
        base64_face1 = _encode_image(face1_path)
        base64_face2 = _encode_image(face2_path)
        
        prompt_text = (
            "Compare ces deux images de visages et détermine si c'est la même personne.\n\n"
            "Analyse les points suivants:\n"
            "1. Les traits du visage (forme, taille)\n"
            "2. Les caractéristiques distinctives (cicatrices, marques, accessoires)\n"
            "3. La structure faciale globale\n"
            "4. Les yeux, nez, bouche, menton\n\n"
            "Réponds en JSON avec cette structure:\n"
            "{\n"
            '  "same_person": true/false,\n'
            '  "confidence": "high"/"medium"/"low",\n'
            '  "reason": "explication brève",\n'
            '  "details": "détails de l\'analyse"\n'
            "}\n\n"
            "Sois prudent et objectif dans ta comparaison."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "text",
                    "text": "Image 1:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_face1}"
                    },
                },
                {
                    "type": "text",
                    "text": "Image 2:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_face2}"
                    },
                },
            ],
        )
        
        response = llm.invoke([message])
        raw_output = response.content.strip()
        
        # Parser la réponse JSON
        import json
        try:
            # Chercher le JSON dans la réponse
            start_idx = raw_output.find('{')
            end_idx = raw_output.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = raw_output[start_idx:end_idx]
                result = json.loads(json_str)
                return result
        except:
            pass
        
        # Si pas de JSON valide, retourner une analyse par défaut
        return {
            "same_person": False,
            "confidence": "low",
            "reason": "Impossible de parser la réponse",
            "details": raw_output
        }
    
    except Exception as e:
        return {
            "same_person": False,
            "confidence": "low",
            "reason": f"Erreur lors de la comparaison: {str(e)}",
            "details": ""
        }


def identify_unique_persons(face_paths: List[str], ollama_base_url: str = None) -> Dict[str, Any]:
    """
    Groupe les visages pour identifier les personnes uniques.
    
    Args:
        face_paths: Liste des chemins des visages
        ollama_base_url: URL de base Ollama
    
    Returns:
        {
            "unique_persons": [{"person_id": int, "face_indices": [int, ...], ...}],
            "total_detections": int,
            "unique_count": int
        }
    """
    if len(face_paths) < 2:
        return {
            "unique_persons": [{"person_id": 0, "face_indices": [0]}],
            "total_detections": len(face_paths),
            "unique_count": 1
        }
    
    # Grouper les visages similaires
    groups = []  # Liste de groupes, chaque groupe contient les indices des visages similaires
    
    for i, face_path1 in enumerate(face_paths):
        assigned = False
        
        # Comparer avec les groupes existants
        for group in groups:
            # Comparer avec le premier visage du groupe
            ref_face_idx = group[0]
            ref_face_path = face_paths[ref_face_idx]
            
            comparison = compare_faces(face_path1, ref_face_path, ollama_base_url)
            
            if comparison.get("same_person") and comparison.get("confidence") in ["high", "medium"]:
                group.append(i)
                assigned = True
                break
        
        if not assigned:
            groups.append([i])
    
    # Créer le résultat
    unique_persons = []
    for person_id, group in enumerate(groups):
        unique_persons.append({
            "person_id": person_id,
            "face_indices": group,
            "detection_count": len(group)
        })
    
    return {
        "unique_persons": unique_persons,
        "total_detections": len(face_paths),
        "unique_count": len(groups)
    }
