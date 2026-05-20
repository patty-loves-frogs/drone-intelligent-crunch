import json
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


REPORTER_MODEL = "qwen2.5:3b"


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

def _risk_score(frame: Dict[str, Any]) -> int:
    score = 0

    for det in frame.get("detections", []):
        posture = det.get("posture")
        risk = det.get("risk_level")
        uncertainty = float(det.get("uncertainty_score", 1.0) or 1.0)
        conf = float(det.get("yolo_confidence", 0.0) or 0.0)

        if posture == "ALLONGE":
            score += 100
        elif posture == "INCERTAIN":
            score += 60
        elif posture == "DEBOUT":
            score += 10

        if risk == "ELEVE":
            score += 50
        elif risk == "A_VERIFIER":
            score += 30

        score += int(conf * 20)
        score += int((1 - uncertainty) * 10)

    return score


def _event_risk_label(frame: Dict[str, Any]) -> str:
    detections = frame.get("detections", [])

    if any(d.get("posture") == "ALLONGE" for d in detections):
        return "ÉLEVÉ"

    if any(d.get("posture") == "INCERTAIN" for d in detections):
        return "À VÉRIFIER"

    return "FAIBLE"


def build_yolo_events_summary(yolo_json: Dict[str, Any], max_frames: int = 8) -> str:
    frames = yolo_json.get("frames", []) if isinstance(yolo_json, dict) else []

    if not frames:
        return "Aucun événement/frame sauvegardé par YOLO."

    ranked_frames = sorted(frames, key=_risk_score, reverse=True)
    selected_frames = ranked_frames[:max_frames]

    lines: List[str] = []

    lines.append(
        "Cette section présente les événements les plus importants détectés dans la vidéo, "
        "classés par niveau de criticité. Les images annotées sont privilégiées comme preuves visuelles."
    )
    lines.append("")

    for idx, frame in enumerate(selected_frames, start=1):
        event_id = frame.get("event_id", "N/A")
        frame_index = frame.get("frame_index", "N/A")
        timestamp = _format_time(frame.get("timestamp_sec"))

        annotated_image_path = frame.get("annotated_image_path", "non précisé")
        raw_image_path = frame.get("raw_image_path", frame.get("image_path", "non précisé"))

        detections = frame.get("detections", [])
        frame_summary = frame.get("summary", {})

        risk_label = _event_risk_label(frame)

        if risk_label == "ÉLEVÉ":
            decision = "Vérification humaine prioritaire"
        elif risk_label == "À VÉRIFIER":
            decision = "Contrôle humain recommandé"
        else:
            decision = "Pas d’alerte critique immédiate"

        lines.append(f"### Événement prioritaire #{idx} — événement {event_id}")
        lines.append(f"**Temps vidéo :** {timestamp}  ")
        lines.append(f"**Frame :** {frame_index}  ")
        lines.append(f"**Niveau de risque :** {risk_label}  ")
        lines.append(f"**Décision recommandée :** {decision}")
        lines.append("")

        lines.append("#### Observations IA")
        lines.append(f"- Personnes détectées : {frame_summary.get('persons_detected', len(detections))}")
        lines.append(f"- Personnes allongées : {frame_summary.get('allonge_count', 0)}")
        lines.append(f"- Personnes debout : {frame_summary.get('debout_count', 0)}")
        lines.append(f"- Postures incertaines : {frame_summary.get('incertain_count', 0)}")
        lines.append(f"- Analyse VLM recommandée : {'oui' if frame_summary.get('needs_vlm', False) else 'non'}")
        lines.append("")

        lines.append("#### Preuve visuelle")
        lines.append(f"- Image annotée : `{annotated_image_path}`")
        lines.append(f"- Image brute originale : `{raw_image_path}`")
        lines.append("")

        lines.append("#### Détails des détections principales")

        for det in detections:
            posture = det.get("posture", "N/A")
            risk = det.get("risk_level", "N/A")
            uncertainty = det.get("uncertainty_score", "N/A")
            conf = det.get("yolo_confidence", det.get("confidence", "N/A"))
            position = det.get("position_in_image", "N/A")

            try:
                conf = round(float(conf), 2)
            except Exception:
                pass

            lines.append(
                f"- Personne #{det.get('id', det.get('person_id', 'N/A'))} : "
                f"posture **{posture}**, risque **{risk}**, "
                f"confiance YOLO **{conf}**, incertitude **{uncertainty}**, "
                f"position **{position}**."
            )

        lines.append("")

        if risk_label == "ÉLEVÉ":
            lines.append(
                "**Interprétation :** YOLO détecte une posture compatible avec une personne au sol. "
                "Cette détection doit être considérée comme prioritaire, même si une validation humaine reste nécessaire."
            )
        elif risk_label == "À VÉRIFIER":
            lines.append(
                "**Interprétation :** la posture détectée est ambiguë. "
                "Le système recommande une vérification complémentaire, notamment via le VLM ou un opérateur humain."
            )
        else:
            lines.append(
                "**Interprétation :** les personnes détectées semblent principalement debout. "
                "Aucune posture critique évidente n’est identifiée sur cette frame."
            )

        lines.append("")

    remaining = len(frames) - len(selected_frames)

    if remaining > 0:
        lines.append("### Autres événements")
        lines.append(
            f"{remaining} frame(s) ou événement(s) supplémentaires ont été détectés, "
            "mais ne sont pas détaillés ici afin de conserver un rapport lisible."
        )

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
        llm = ChatOllama(model=REPORTER_MODEL, temperature=0.2)
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
