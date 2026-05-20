import base64
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def _encode_image(image_path: str) -> str:
    """Encode une image en base64 pour l'envoyer au VLM."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image_with_vlm(image_path: str, instruction: str) -> Dict[str, Any]:
    """
    Analyse une image avec un VLM local via Ollama (modèle recommandé : llava).

    Retourne un dictionnaire structuré :
      - image
      - person_visible (bool)
      - description (str)
      - confidence ('low' | 'medium' | 'high')
      - reason (str)
    """
    llm = ChatOllama(model="mistral", temperature=0.2, base_url="http://192.168.0.104:11434")
    base64_image = _encode_image(image_path)

    prompt_text = (
        f"L'utilisateur demande : '{instruction}'\n\n"
        "Analyse cette image et réponds en français aux questions suivantes :\n"
        "1. Y a-t-il une personne dans l'image ? (oui/non)\n"
        "2. Où est-elle approximativement ?\n"
        "3. Est-elle debout, assise, couchée ou difficile à distinguer ?\n"
        "4. Niveau de confiance : low, medium, high ?\n"
        "Réponds de manière concise et structurée."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        ],
    )

    response = llm.invoke([message])
    raw_output = response.content.strip()

    # Heuristique simple pour extraire la confiance
    text_lower = raw_output.lower()
    person_visible = "oui" in text_lower or "yes" in text_lower
    confidence = "low"
    if "high" in text_lower or "élevé" in text_lower or "forte" in text_lower:
        confidence = "high"
    elif "medium" in text_lower or "moyen" in text_lower or "moyenne" in text_lower:
        confidence = "medium"

    return {
        "image": image_path,
        "person_visible": person_visible,
        "description": raw_output,
        "confidence": confidence,
        "reason": "VLM analysis triggered by YOLO uncertainty or absence of person detection.",
    }


def vlm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nœud VLM :
      - Lit les vlm_candidates du state (produits par YOLO).
      - Appelle analyze_image_with_vlm pour chaque image candidate.
      - Retourne vlm_observations.
    """
    vlm_candidates: List[str] = state.get("vlm_candidates", [])
    instruction: str = state.get("instruction", "")
    vlm_observations: List[Dict[str, Any]] = []

    if not vlm_candidates:
        return {"vlm_observations": vlm_observations}

    for img_path in vlm_candidates:
        try:
            vlm_result = analyze_image_with_vlm(img_path, instruction)
            vlm_observations.append(vlm_result)
        except Exception as e:
            vlm_observations.append({
                "image": img_path,
                "person_visible": False,
                "description": f"VLM error: {e}",
                "confidence": "low",
                "reason": "VLM call failed after YOLO uncertainty.",
            })

    return {"vlm_observations": vlm_observations}