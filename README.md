# Drone Intelligent Conversationnel (MVP)

Prototype de drone intelligent conversationnel capable d'analyser une vidéo drone, de détecter des situations potentiellement critiques, puis de générer un rapport de mission exploitable par un opérateur humain.

Le système combine :

- **YOLOv8 Pose** pour détecter les personnes et estimer leur posture ;
- **VLM local via Ollama** pour analyser les images critiques ou ambiguës ;
- **LLM Reporter local via Ollama** pour produire un rapport opérationnel ;
- **Streamlit** pour l'interface utilisateur et le dashboard mission.

L'objectif principal est d'aider un opérateur à repérer rapidement une personne potentiellement au sol dans une vidéo drone.

---

## Stack technique

- Python 3.11+
- Streamlit
- LangGraph
- LangChain
- Ollama
- Ultralytics YOLOv8
- OpenCV
- Pandas
- OpenAI Python SDK compatible Ollama

---

## Architecture du projet

```text
drone-intelligent/
│
├── app.py                  # Interface Streamlit / dashboard mission
├── graph.py                # Pipeline LangGraph et noeuds YOLO / VLM
├── yolo.py                 # Analyse vidéo YOLOv8 Pose + export JSON
├── vlm_call.py             # Appel VLM local via Ollama / API compatible OpenAI
├── reporter.py             # Génération du rapport mission Markdown
│
├── videos/                 # Vidéos drone à analyser
├── weights/                # Poids YOLO ou modèles locaux éventuels
├── outputs/
│   └── runs/
│       └── <run_id>/
│           ├── raw/        # Frames brutes extraites de la vidéo
│           ├── annotated/  # Frames annotées par YOLO
│           ├── analysis.json
│           └── rapport_mission.md
│
├── requirements.txt
└── README.md
```

---

## Pipeline IA

```text
Vidéo drone
    ↓
YOLOv8 Pose
    ↓
Détection des personnes et estimation de posture
    ↓
Sélection automatique des frames critiques
    ↓
VLM local pour les cas critiques ou incertains
    ↓
Reporter LLM
    ↓
Dashboard Streamlit + rapport Markdown
```

Le pipeline actuel est volontairement simple :

```text
yolo -> vlm -> reporter -> END
```

Il n'y a pas de noeud planner.

---

## Fonctionnalités principales

- Analyse d'une vidéo drone au format `.mp4`, `.mov`, `.avi` ou `.mkv`
- Détection de personnes avec YOLOv8 Pose
- Classification simple des postures :
  - `DEBOUT`
  - `ALLONGE`
  - `INCERTAIN`
- Estimation d'un niveau de risque :
  - `FAIBLE`
  - `A_VERIFIER`
  - `ELEVE`
- Sauvegarde des frames brutes et annotées
- Sélection automatique des images candidates pour le VLM
- Analyse complémentaire par VLM local
- Dashboard mission Streamlit
- Timeline des événements détectés
- Rapport de mission généré en Markdown
- Export du rapport `.md`

---

## Prérequis

### 1. Installer Python

Python 3.11 ou supérieur est recommandé.

Vérification :

```bash
python --version
```

### 2. Installer Ollama

Télécharger Ollama :

<https://ollama.com>

Vérifier l'installation :

```bash
ollama list
```

### 3. Télécharger les modèles locaux

#### VLM pour l'analyse d'image

```bash
ollama pull llava
```

#### LLM Reporter pour le rapport

```bash
ollama pull qwen2.5:3b
```

Vérifier que les modèles sont disponibles :

```bash
ollama list
```

Exemple attendu :

```text
NAME            ID              SIZE
llava:latest    ...             ...
qwen2.5:3b      ...             ...
```

> Si `qwen2.5:3b` n'est pas installé, le système peut quand même produire un rapport simplifié grâce au fallback codé dans `reporter.py`.

---

## Installation

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

Si besoin, installer aussi le SDK OpenAI compatible avec Ollama :

```bash
pip install openai
```

---

## Configuration VLM

Le fichier `vlm_call.py` utilise par défaut Ollama via l'API compatible OpenAI :

```python
VLM_BASE_URL = "http://127.0.0.1:11434/v1"
VLM_MODEL = "llava:latest"
VLM_API_KEY = "ollama"
```

Cela signifie que :

- le VLM tourne en local avec Ollama ;
- aucune image n'est envoyée à une API externe ;
- les images critiques sont transmises au VLM sous forme base64.

Pour vérifier que le serveur Ollama répond :

```bash
python -c "from openai import OpenAI; c=OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama'); print(c.models.list())"
```

---

## Lancer l'application

Depuis la racine du projet :

```bash
python -m streamlit run app.py
```

L'interface Streamlit s'ouvre ensuite dans le navigateur.

---

## Utilisation

1. Placer une vidéo dans le dossier `videos/`.
2. Lancer l'application Streamlit.
3. Dans le champ **Chemin de la vidéo à analyser**, renseigner par exemple :

```text
videos/mission.mp4
```

4. Dans le champ **Instruction / Mission**, renseigner par exemple :

```text
Dis-moi s'il y a quelqu'un au sol dans la vidéo.
```

5. Cliquer sur **Analyser**.

Le système exécute alors :

1. l'analyse YOLO de la vidéo ;
2. la sélection des frames critiques ;
3. l'analyse VLM des cas nécessaires ;
4. la génération du rapport de mission.

---

## Sorties générées

Chaque analyse crée un nouveau dossier dans :

```text
outputs/runs/
```

Exemple :

```text
outputs/runs/20260520_221642_DJI_20260518150714_0002_V/
```

Ce dossier contient :

```text
raw/                 # Images brutes extraites de la vidéo
annotated/           # Images annotées avec les détections YOLO
analysis.json        # Données complètes de l'analyse
rapport_mission.md   # Rapport final exportable
```

---

## Interface Streamlit

Le dashboard affiche :

- le statut de progression de l'analyse ;
- le niveau de risque global ;
- le nombre de personnes détectées ;
- le nombre d'événements détectés ;
- le nombre de personnes allongées ;
- le nombre de postures incertaines ;
- les images annotées ;
- la timeline des événements ;
- les résultats VLM ;
- le rapport final.

Les images affichées dans l'interface sont les images annotées présentes dans :

```text
outputs/runs/<run_id>/annotated/
```

Le VLM reçoit quant à lui les images brutes présentes dans :

```text
outputs/runs/<run_id>/raw/
```

Ce choix permet :

- d'afficher à l'utilisateur des preuves visuelles compréhensibles ;
- de laisser le VLM analyser l'image originale sans surcouche graphique.

---

## Rapport de mission

Le rapport Markdown contient notamment :

1. l'instruction utilisateur ;
2. la synthèse opérationnelle ;
3. le résumé vidéo / YOLO ;
4. les événements critiques détectés ;
5. l'analyse VLM ;
6. la recommandation opérationnelle ;
7. les limites du système.

Le rapport met en avant :

- les événements les plus critiques ;
- les timestamps importants ;
- les images annotées associées ;
- les niveaux de risque ;
- la décision recommandée.

---

## Gestion des erreurs

### Vidéo introuvable

L'interface affiche une erreur si le chemin vidéo n'existe pas.

### Aucun événement détecté

Le rapport indique qu'aucune personne ou situation critique n'a été détectée.

### VLM indisponible

Si Ollama ou le modèle VLM ne répond pas, le pipeline continue avec les résultats YOLO disponibles.

### Reporter LLM indisponible

Si le modèle reporter n'est pas installé ou ne répond pas, un rapport simplifié est généré automatiquement par le fallback Python.

---

## Conseils de performance

Le modèle `llava:latest` peut être lent sur CPU.

Pour accélérer la démonstration :

- limiter le nombre d'images envoyées au VLM ;
- conserver uniquement les frames les plus critiques ;
- utiliser une machine avec GPU si possible ;
- augmenter le stride d'analyse YOLO si la vidéo est longue.

Dans `graph.py`, le VLM peut être limité avec :

```python
vlm_candidates = vlm_candidates[:3]
```

---

## Valeur opérationnelle

Ce projet vise à assister des opérateurs drone dans des contextes comme :

- inspection de zone ;
- surveillance ;
- sécurité ;
- recherche de personne ;
- situation post-incident.

La valeur ajoutée principale :

- réduction du temps de revue vidéo ;
- priorisation automatique des événements critiques ;
- génération d'un rapport lisible ;
- fonctionnement local sans API payante ;
- conservation des données sensibles sur la machine de l'utilisateur.

Le système ne remplace pas l'opérateur humain : il sert d'outil d'aide à la décision.

---

## Limites connues

- YOLO Pose peut se tromper sur les postures lorsque la personne est partiellement visible.
- Le VLM peut halluciner ou manquer des détails sur des images complexes.
- Les résultats doivent être validés par un humain dans un contexte critique.
- Les performances dépendent fortement de la machine utilisée.
- L'analyse vidéo peut être longue sur CPU.

---

## Commandes utiles

### Lancer l'application

```bash
python -m streamlit run app.py
```

### Vérifier les modèles Ollama

```bash
ollama list
```

### Télécharger le VLM

```bash
ollama pull llava
```

### Télécharger le reporter

```bash
ollama pull qwen2.5:3b
```

### Tester l'API Ollama compatible OpenAI

```bash
python -c "from openai import OpenAI; c=OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama'); print(c.models.list())"
```

---

## Contraintes respectées

- Fonctionnement local
- Pas d'API payante
- Pas d'envoi d'images sensibles à un service externe
- Pipeline simple et explicable
- Rapport exportable
- Interface présentable pour une démonstration

