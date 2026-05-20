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

### Configuration Ollama

1. Copiez le fichier `.env.example` en `.env` :
   ```bash
   cp .env.example .env
   ```

2. Configurez l'URL de votre instance Ollama dans `.env` :
   ```env
   # Localhost (défaut)
   OLLAMA_BASE_URL=http://localhost:11434
   
   # Instance distante
   OLLAMA_BASE_URL=http://192.168.1.100:11434
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


## Llama.CPP
Pour pouvoir envoyer des requêtes aux modèles, il faut télécharger **Llama.cpp** : https://github.com/ggml-org/llama.cpp (https://github.com/ggml-org/llama.cpp/releases)
Télécharger la version qui correspond au matériel (CUDA pour GPU NVIDIA).

Dans le cas présent, les modèles chargés par le serveur se trouveront dans ce dossier : 
"C:\mes-modeles\MODELE_CHARGE\"

Lancer un serveur par modèle à l'aide de cette commande : 
"C:\[...]\mon-serveur-llama\llama-server.exe" ^ 
--models-dir "C:\[...]\drone-intelligent-crunch\weights\" ^ 
--port 8080 ^ 
--parallel 1 ^ 
--reasoning-budget 0 ^ 
--reasoning off ^ 
--n-gpu-layers 256 ^ 
--cache-prompt ^ 
--image-max-tokens 1024 ^
--ctx-size 65536
-fa on ^
--spec-type draft-mtp ^
--spec-draft-n-max 6 ^
--no-ui

Reasoning budget est désactivé car l'objectif du VLM ici est d'être rapide et performant.
Parallel est désactivé pour n'avoir qu'un seul slot car utilisateur unique.
Image max tokens est mis à 1024 car modèle Qwen3.5-4B-Q4_K_M utilisé.
Adapter le contexte selon les besoins.

## Modèle (VLM)
Télécharger les poids VLM du modèle qui vous convient et les déposer dans un nouveau dossier dans "/weights/VLM" (GGUF & mmproj)
Pour Qwen3.5 2B MTP GGUF : https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF/tree/main
