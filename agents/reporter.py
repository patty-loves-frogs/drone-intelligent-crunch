from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def reporter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nœud reporter :
      - Agrège les détections YOLO et les observations VLM.
      - Génère un rapport final structuré via un LLM local (Ollama / qwen2.5:3b).
      - En cas d'échec du LLM, produit un rapport basique automatique.
    """
    instruction: str = state.get("instruction", "")
    images: list[str] = state.get("images", [])
    detections: list[dict] = state.get("detections", [])
    vlm_observations: list[dict] = state.get("vlm_observations", [])

    if not images:
        return {
            "report": "Aucune image trouvée dans le dossier frames/. "
            "Veuillez ajouter des images avant de lancer l'analyse."
        }

    # Résumé YOLO (exclure les erreurs du résumé principal)
    yolo_lines = []
    for d in detections:
        if d.get("label") == "error":
            continue
        yolo_lines.append(
            f"- {d['image']}: {d['label']} (conf: {d['confidence']}, bbox: {d['bounding_box']})"
        )
    yolo_summary = "\n".join(yolo_lines) if yolo_lines else "Aucune détection YOLO."

    # Résumé VLM
    vlm_lines = []
    for obs in vlm_observations:
        vlm_lines.append(
            f"- {obs['image']}: person_visible={obs['person_visible']}, "
            f"confidence={obs['confidence']}, desc={obs['description']}"
        )
    vlm_summary = "\n".join(vlm_lines) if vlm_lines else "Aucune analyse VLM effectuée."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un agent de rapport pour un drone de surveillance. "
                "Rédige un rapport structuré en français basé sur les données fournies.",
            ),
            (
                "human",
                (
                    "Instruction utilisateur : {instruction}\n\n"
                    "Images analysées : {images_count}\n\n"
                    "Résultats YOLO :\n{yolo_summary}\n\n"
                    "Résultats VLM :\n{vlm_summary}\n\n"
                    "Génère un rapport final structuré contenant :\n"
                    "- Résumé\n"
                    "- Images analysées\n"
                    "- Résultats YOLO\n"
                    "- Résultats VLM si utilisés\n"
                    "- Personnes détectées\n"
                    "- Niveau de confiance global\n"
                    "- Incertitudes\n"
                    "- Recommandation opérationnelle\n"
                ),
            ),
        ]
    )

    try:
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
        chain = prompt_template | llm
        response = chain.invoke(
            {
                "instruction": instruction,
                "images_count": len(images),
                "yolo_summary": yolo_summary,
                "vlm_summary": vlm_summary,
            }
        )
        report = response.content
    except Exception as e:
        report = (
            f"## RAPPORT BASIQUE (LLM indisponible : {e})\n\n"
            f"**Instruction :** {instruction}\n\n"
            f"**Images analysées :** {len(images)}\n\n"
            f"**Détections YOLO :**\n{yolo_summary}\n\n"
            f"**Observations VLM :**\n{vlm_summary}\n\n"
            f"**Recommandation :** Vérifier manuellement les images concernées."
        )

    return {"report": report}
