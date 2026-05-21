from pathlib import Path

reporter_code = r'''import os
import json
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


REPORTER_MODEL = os.getenv("REPORTER_MODEL", "qwen2.5:3b")
REPORTER_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ============================================================
# OUTILS
# ============================================================

def _safe_get(d: Dict[str, Any], key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _truncate(text: str, max_chars: int = 8000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[contenu tronqué]"


def _format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "temps inconnu"

    try:
        seconds = float(seconds)
    except Exception:
        return "temps inconnu"

    minutes = int(seconds // 60)
    sec = seconds % 60

    if minutes > 0:
        return f"{minutes} min {sec:.1f} s"

    return f"{sec:.1f} s"


def _parse_vlm_text_as_json(text: str) -> Dict[str, Any]:
    """
    Certains VLM retournent du JSON sous forme de string.
    Cette fonction essaie de le parser, mais ne casse pas le pipeline si ce n'est pas du JSON.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    raw = text.strip()

    # Nettoyage basique si le modèle renvoie ```json ... ```
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


# ============================================================
# RESUMES YOLO
# ============================================================

def build_video_summary(yolo_json: Dict[str, Any]) -> str:
    summary = yolo_json.get("summary", {}) if isinstance(yolo_json, dict) else {}
    output_paths = yolo_json.get("output_paths", {}) if isinstance(yolo_json, dict) else {}
    parameters = yolo_json.get("parameters", {}) if isinstance(yolo_json, dict) else {}

    lines = [
        f"Vidéo analysée : {yolo_json.get('video_path', 'non précisée')}",
        f"Run : {yolo_json.get('run_name', 'non précisé')}",
        f"JSON YOLO : {output_paths.get('json_path', 'non précisé')}",
        f"Frames / événements conservés : {yolo_json.get('frames_analyzed', 0)}",
        f"Modèle YOLO : {parameters.get('model', 'non précisé')}",
        f"Stride d'analyse : {parameters.get('analysis_stride', 'non précisé')}",
        "",
        "Résumé détections :",
        f"- Personnes détectées : {summary.get('persons_detected', 0)}",
        f"- Personnes debout : {summary.get('standing_count', 0)}",
        f"- Personnes allongées : {summary.get('lying_count', 0)}",
        f"- Postures incertaines : {summary.get('uncertain_count', 0)}",
        f"- Candidats VLM : {summary.get('vlm_candidate_count', 0)}",
        f"- Événements détectés : {summary.get('events_count', 0)}",
    ]

    error = yolo_json.get("error")
    if error:
        lines.append(f"Erreur YOLO : {error}")

    return "\n".join(lines)


def build_yolo_events_summary(yolo_json: Dict[str, Any], max_frames: int = 20) -> str:
    frames = yolo_json.get("frames", []) if isinstance(yolo_json, dict) else []

    if not frames:
        return "Aucun événement/frame sauvegardé par YOLO."

    lines: List[str] = []

    for frame in frames[:max_frames]:
        event_id = frame.get("event_id", "N/A")
        frame_index = frame.get("frame_index", "N/A")
        timestamp = _format_time(frame.get("timestamp_sec"))
        raw_image_path = frame.get("raw_image_path", frame.get("image_path", "non précisé"))
        annotated_image_path = frame.get("annotated_image_path", "non précisé")
        detections = frame.get("detections", [])
        frame_summary = frame.get("summary", {})

        lines.append(
            f"Événement {event_id} | frame {frame_index} | t={timestamp}\n"
            f"- Image brute : {raw_image_path}\n"
            f"- Image annotée : {annotated_image_path}\n"
            f"- Résumé frame : personnes={frame_summary.get('persons_detected', len(detections))}, "
            f"allongées={frame_summary.get('allonge_count', 0)}, "
            f"debout={frame_summary.get('debout_count', 0)}, "
            f"incertaines={frame_summary.get('incertain_count', 0)}, "
            f"needs_vlm={frame_summary.get('needs_vlm', False)}"
        )

        for det in detections:
            lines.append(
                f"  • personne#{det.get('id', det.get('person_id', 'N/A'))} | "
                f"posture={det.get('posture', 'N/A')} | "
                f"risque={det.get('risk_level', 'N/A')} | "
                f"incertitude={det.get('uncertainty_score', 'N/A')} | "
                f"conf_yolo={det.get('yolo_confidence', det.get('confidence', 'N/A'))} | "
                f"position={det.get('position_in_image', 'N/A')} | "
                f"bbox={det.get('bbox_xyxy', det.get('bounding_box', 'N/A'))}"
            )

        lines.append("")

    if len(frames) > max_frames:
        lines.append(f"... {len(frames) - max_frames} frame(s)/événement(s) non affiché(s).")

    return "\n".join(lines)


# ============================================================
# RESUMES VLM
# ============================================================

def build_vlm_summary(vlm_observations: List[Dict[str, Any]]) -> str:
    if not vlm_observations:
        return "Aucune analyse VLM effectuée."

    lines: List[str] = []

    for idx, obs in enumerate(vlm_observations, start=1):
        image_path = (
            obs.get("image_path")
            or obs.get("image")
            or obs.get("raw_image_path")
            or "image non précisée"
        )

        error = obs.get("error")
        text = (
            obs.get("text")
            or obs.get("description")
            or obs.get("raw_output")
            or ""
        )

        parsed = _parse_vlm_text_as_json(text)

        person_visible = obs.get("person_visible", parsed.get("person_count", "N/A"))
        confidence = obs.get("confidence", "N/A")

        lines.append(f"VLM #{idx} — image : {image_path}")

        if error:
            lines.append(f"- Erreur : {error}")
            continue

        if parsed:
            lines.append(f"- Description : {parsed.get('desc', 'N/A')}")
            lines.append(f"- Nombre de personnes : {parsed.get('person_count', 'N/A')}")
            lines.append(f"- Localisations : {parsed.get('locs', [])}")
            lines.append(f"- Situation non standard : {parsed.get('non_std', 'N/A')}")
        else:
            lines.append(f"- Personne visible : {person_visible}")
            lines.append(f"- Confiance : {confidence}")
            lines.append(f"- Texte : {text}")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# RAPPORT BASIQUE FALLBACK
# ============================================================

def build_basic_report(
    instruction: str,
    yolo_json: Dict[str, Any],
    yolo_summary: str,
    yolo_events_summary: str,
    vlm_summary: str,
    llm_error: Optional[Exception] = None,
) -> str:
    summary = yolo_json.get("summary", {}) if isinstance(yolo_json, dict) else {}

    persons = summary.get("persons_detected", 0)
    lying = summary.get("lying_count", 0)
    uncertain = summary.get("uncertain_count", 0)
    vlm_candidates = summary.get("vlm_candidate_count", 0)

    if lying > 0:
        risk = "ÉLEVÉ"
        recommendation = (
            "Prioriser une vérification humaine rapide des frames où une personne est détectée allongée."
        )
    elif uncertain > 0 or vlm_candidates > 0:
        risk = "À VÉRIFIER"
        recommendation = (
            "Vérifier manuellement les frames incertaines et comparer les résultats YOLO/VLM."
        )
    elif persons > 0:
        risk = "FAIBLE À MODÉRÉ"
        recommendation = (
            "Présence humaine détectée, sans posture critique évidente selon YOLO."
        )
    else:
        risk = "FAIBLE"
        recommendation = (
            "Aucune personne détectée dans les événements conservés. Vérifier la vidéo si la mission est critique."
        )

    llm_note = ""
    if llm_error:
        llm_note = f"\n\n> Rapport généré sans LLM reporter, erreur : `{llm_error}`\n"

    return f"""# Rapport mission drone

## 1. Instruction utilisateur
{instruction or "Instruction non précisée."}

## 2. Synthèse opérationnelle
- Niveau de risque estimé : **{risk}**
- Personnes détectées : **{persons}**
- Personnes allongées : **{lying}**
- Postures incertaines : **{uncertain}**
- Frames candidates VLM : **{vlm_candidates}**

## 3. Résumé vidéo / YOLO
{yolo_summary}

## 4. Événements détectés par YOLO
{yolo_events_summary}

## 5. Analyse VLM
{vlm_summary}

## 6. Recommandation
{recommendation}

## 7. Limites
- YOLO Pose estime la posture à partir de keypoints et de ratios de bounding box.
- Le VLM peut aider sur les cas ambigus, mais peut aussi halluciner.
- Une validation humaine reste nécessaire pour les situations de secours ou de sécurité.
{llm_note}
"""


# ============================================================
# NODE LANGGRAPH
# ============================================================

def reporter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent reporter.

    Entrée attendue dans le state :
    - instruction
    - yolo_json
    - detections
    - vlm_observations

    Sortie :
    - report : texte Markdown
    """

    instruction: str = state.get("instruction", "")
    yolo_json: Dict[str, Any] = state.get("yolo_json", {})
    vlm_observations: List[Dict[str, Any]] = state.get("vlm_observations", [])

    yolo_summary = build_video_summary(yolo_json)
    yolo_events_summary = build_yolo_events_summary(yolo_json)
    vlm_summary = build_vlm_summary(vlm_observations)

    if not yolo_json or yolo_json.get("error"):
        report = build_basic_report(
            instruction=instruction,
            yolo_json=yolo_json,
            yolo_summary=yolo_summary,
            yolo_events_summary=yolo_events_summary,
            vlm_summary=vlm_summary,
        )
        return {
            **state,
            "report": report,
        }

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Tu es l'agent de rapport d'un drone intelligent conversationnel. "
                    "Tu dois produire un rapport opérationnel clair, structuré, en français. "
                    "Tu ne dois pas inventer de faits absents des données. "
                    "Tu dois distinguer les certitudes YOLO, les incertitudes, et les confirmations VLM."
                ),
            ),
            (
                "human",
                (
                    "Instruction utilisateur :\n{instruction}\n\n"
                    "Résumé global YOLO :\n{yolo_summary}\n\n"
                    "Détails événements YOLO :\n{yolo_events_summary}\n\n"
                    "Résultats VLM :\n{vlm_summary}\n\n"
                    "Génère un rapport Markdown avec exactement ces sections :\n"
                    "1. Résumé exécutif\n"
                    "2. Situation observée\n"
                    "3. Événements et preuves visuelles\n"
                    "4. Analyse des risques\n"
                    "5. Incertitudes et limites\n"
                    "6. Recommandation opérationnelle\n\n"
                    "Contraintes :\n"
                    "- Mentionne les timestamps ou frames quand ils existent.\n"
                    "- Mentionne les images sauvegardées utiles.\n"
                    "- Si une personne est ALLONGE ou non_std=yes côté VLM, signale une priorité de vérification.\n"
                    "- Si les données sont insuffisantes, dis-le explicitement.\n"
                ),
            ),
        ]
    )

    try:
        llm = ChatOllama(model=REPORTER_MODEL, base_url=REPORTER_BASE_URL, temperature=0.2)
        chain = prompt_template | llm

        response = chain.invoke(
            {
                "instruction": instruction,
                "yolo_summary": _truncate(yolo_summary, 4000),
                "yolo_events_summary": _truncate(yolo_events_summary, 9000),
                "vlm_summary": _truncate(vlm_summary, 5000),
            }
        )

        report = response.content

    except Exception as e:
        report = build_basic_report(
            instruction=instruction,
            yolo_json=yolo_json,
            yolo_summary=yolo_summary,
            yolo_events_summary=yolo_events_summary,
            vlm_summary=vlm_summary,
            llm_error=e,
        )

    return {
        **state,
        "report": report,
    }
'''

path = Path("/mnt/data/reporter.py")
path.write_text(reporter_code, encoding="utf-8")
print(f"Fichier créé : {path}")
