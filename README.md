# Drone Intelligent Conversationnel (MVP)

Prototype open source d'un drone intelligent conversationnel basé sur **LangGraph**, **YOLOv8**, **VLM (Ollama)** et **LLM Reporter (Ollama)**.

## Stack technique

- Python 3.11+
- LangGraph + LangChain
- Ollama (locale)
- Ultralytics YOLOv8
- OpenCV
- Streamlit

## Architecture

```
drone-intelligent/
  app.py                  # Interface Streamlit
  graph.py                # Graphe LangGraph
  agents/
    __init__.py
    yolo.py               # Détection YOLOv8 + sélection des candidates VLM
    vlm.py                # Analyse image par VLM (Ollama / llava)
    reporter.py           # Génération du rapport final (Ollama / qwen2.5)
  frames/                 # Images du drone à analyser
  outputs/                # Sorties optionnelles
  requirements.txt
  README.md
```

## Graphe LangGraph

```
yolo -> vlm -> reporter -> END
```

Pas de nœud planner.

## Prérequis

1. **Installer Ollama** : [https://ollama.com](https://ollama.com)
2. **Télécharger les modèles locaux** :
   ```bash
   ollama pull qwen2.5:3b
   ollama pull llava
   ```

## Installation

```bash
pip install -r requirements.txt
```

> **Note** : la première exécution téléchargera automatiquement `yolov8n.pt` (YOLOv8 nano) via Ultralytics.

## Lancer l'application

Placez vos images (`.jpg`, `.jpeg`, `.png`) dans le dossier `frames/`, puis :

```bash
python -m streamlit run app.py
```

## Utilisation

1. Ouvrez l'interface Streamlit dans votre navigateur.
2. Saisissez une instruction, par exemple :
   > *"Dis-moi s'il y a quelqu'un au sol dans les images."*
3. Cliquez sur **Analyser**.
4. Consultez les images, les détections YOLO, les éventuelles analyses VLM et le rapport final.

## Gestion des erreurs

- **Aucune image** : le rapport indique *"Aucune image trouvée dans le dossier frames/"*.
- **YOLO échoue** : le système bascule sur le VLM si possible.
- **VLM échoue** : le rapport est généré avec les résultats YOLO disponibles.
- **LLM Reporter échoue** : un rapport basique est généré manuellement sans LLM.

## Contraintes respectées

- Aucune API payante (OpenAI, etc.).
- Tout fonctionne localement via Ollama.
- Pas de nœud planner dans le graphe.
