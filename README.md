# 🌍 FasoConnect — API Linguistique Burkinabè

API de traduction automatique et synthèse vocale pour les langues du Burkina Faso, construite avec **Meta NLLB-200** et **Meta MMS-TTS**, servie via **FastAPI** et **Docker**.

---

## 📋 Table des matières

- [Langues supportées](#-langues-supportées)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Lancer l'API](#-lancer-lapi)
- [Endpoints](#-endpoints)
- [Historique](#-historique)
- [Tester avec Postman](#-tester-avec-postman)
- [Écouter l'audio TTS](#-écouter-laudio-tts)
- [Fine-tuning](#-fine-tuning)
- [Structure du projet](#-structure-du-projet)
- [Performances](#-performances)
- [Dépannage](#-dépannage)

---

## 🗣️ Langues supportées

| Langue | Code NLLB | Code interne | Traduction | TTS | Fine-tuning |
|---|---|---|---|---|---|
| Mooré | `mos_Latn` | `moore` | ✅ | ✅ | ✅ |
| Dioula / Jula | `dyu_Latn` | `dioula` | ✅ | ✅ | ✅ |
| Fulfulde | `fuv_Latn` | `fulfulde` | ✅ | ✅ | ✅ |
| Gourmantchéma | `gux_Latn` | `gourmantsema` | ✅ | ✅ | ⚠️ peu de données |
| Dagaare | `dga_Latn` | `dagaare` | ✅ | ✅ | ⚠️ peu de données |
| Français | `fra_Latn` | `francais` | ✅ | ✅ | pivot |
| Anglais | `eng_Latn` | `anglais` | ✅ | ✅ | pivot |

Toutes les combinaisons bidirectionnelles sont supportées :
`Français ↔ Langue locale`, `Anglais ↔ Langue locale`, `Langue locale ↔ Langue locale`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│      Clients (Postman · audio_player · Angular)     │
└──────────────────────────┬──────────────────────────┘
                           │ HTTPS + X-API-Key
┌──────────────────────────▼──────────────────────────┐
│   FastAPI — Auth · Rate Limit · CORS · Logs JSON    │
│                                                     │
│  POST /translate          POST /tts                 │
│  POST /translate-and-speak                          │
│  GET/POST/DELETE /history                           │
│                                                     │
│  ┌──────────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ NLLB-200     │  │ MMS-TTS   │  │ SQLite /    │  │
│  │ distilled    │  │ 7 langues │  │ PostgreSQL  │  │
│  │ 600M · CUDA  │  │ CUDA      │  │ + Alembic   │  │
│  └──────────────┘  └───────────┘  └─────────────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │         Redis Cache (TTL 1h)                │   │
│  └─────────────────────────────────────────────┘   │
└──────────── Docker Compose ─────────────────────────┘
              WSL2 · RTX 3060 · CUDA 13.3
```

**Stack technique :**
- `FastAPI` + `Uvicorn` — API async
- `Meta NLLB-200 distilled 600M` — traduction 7 langues
- `Meta MMS-TTS` — synthèse vocale multilingue
- `SQLModel` + `Alembic` — historique des requêtes
- `Redis` — cache traductions et audio (TTL 1h)
- `SlowAPI` — rate limiting par IP
- `Docker Compose` — containerisation complète

---

## ⚙️ Prérequis

| Outil | Version | Notes |
|---|---|---|
| Docker Desktop | 4.x+ | Avec WSL2 backend |
| WSL2 + Ubuntu 22.04 | — | Pour le GPU |
| GPU NVIDIA | Driver 610+ | Optionnel mais fortement recommandé |
| RAM allouée WSL | 10 Go minimum | Voir `.wslconfig` |
| Disque libre | 15 Go minimum | Modèles ~10 Go |

---

## 🚀 Installation

### 1 — Cloner le projet

```bash
git clone https://github.com/assami/fasoconnect.git
cd fasoconnect
```

### 2 — Configurer l'environnement

```bash
cp .env.example .env
# Édite .env avec tes valeurs
```

### 3 — Générer une API Key

```bash
# Dans WSL
openssl rand -hex 32
```

### 4 — Configurer WSL2 (Windows)

Crée ou édite `C:\Users\<user>\.wslconfig` :

```ini
[wsl2]
memory=10GB
processors=8
swap=4GB
gpuSupport=true
networkingMode=mirrored
```

Puis redémarre WSL :
```powershell
wsl --shutdown
```

### 5 — Activer le GPU dans Docker

```bash
# Dans Ubuntu WSL
sudo nvidia-ctk runtime configure --runtime=docker
```

Redémarre Docker Desktop, puis vérifie :
```bash
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

---

## ⚙️ Configuration `.env`

| Variable | Description | Défaut |
|---|---|---|
| `HF_TOKEN` | Token HuggingFace ([créer ici](https://huggingface.co/settings/tokens)) | — |
| `HUGGING_FACE_HUB_TOKEN` | Même valeur que HF_TOKEN (évite un warning) | — |
| `API_KEY` | Clé d'accès à l'API (générer avec `openssl rand -hex 32`) | — |
| `ALLOWED_ORIGIN` | Domaine CORS autorisé | `*` |
| `DATABASE_URL` | URL base de données | `sqlite:///./fasoconnect.db` |
| `CACHE_TTL` | Durée cache Redis en secondes | `3600` |
| `INFER_TIMEOUT` | Timeout inférence en secondes | `60` |

---

## ▶️ Lancer l'API

```powershell
# Premier lancement (~10–20 min — télécharge les modèles)
docker compose up --build -d

# Lancements suivants (~30 sec)
docker compose up -d

# Voir les logs
docker compose logs -f api

# Arrêter
docker compose down
```

**Logs attendus au démarrage :**
```
api-1 | ✅ HuggingFace authentifié
api-1 | ✅ Base de données prête
api-1 | ✅ NLLB-200 chargé sur cuda
api-1 | ✅ MMS-TTS prêt
api-1 | INFO: Uvicorn running on http://0.0.0.0:8000
```

**Swagger UI disponible sur :** `http://localhost:8000/docs`

---

## 📡 Endpoints

### Authentification
Ajouter le header à chaque requête (si `API_KEY` configurée) :
```
X-API-Key: ta_cle_api
```

---

### `GET /health`
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models": ["nllb-200-distilled-600M", "mms-tts"],
  "cache": "redis"
}
```

---

### `GET /languages`
Retourne les langues supportées pour la traduction et le TTS.

---

### `POST /translate`
Rate limit : **20 req/min par IP**

```json
// Body
{ "text": "Bonjour, comment allez-vous ?", "src_lang": "francais", "tgt_lang": "moore" }

// Réponse 200
{ "translated_text": "Ne y welame ?", "src_lang": "francais", "tgt_lang": "moore", "cached": false }
```

---

### `POST /tts`
Rate limit : **10 req/min par IP**

```json
// Body
{ "text": "Ne y welame", "lang": "moore", "speed": 1.0 }

// Réponse 200
{ "audio_b64": "UklGRiQ...", "sample_rate": 16000, "duration_seconds": 2.3, "lang": "moore" }
```

`speed` : `0.5` (lent) → `1.0` (normal) → `2.0` (rapide)

---

### `POST /translate-and-speak`
Rate limit : **10 req/min par IP**

```json
// Body
{ "text": "Bonjour tout le monde", "src_lang": "francais", "tgt_lang": "dioula", "speed": 1.0 }

// Réponse 200
{
  "original_text": "Bonjour tout le monde",
  "translated_text": "I ni ce",
  "audio_b64": "UklGRiQ...",
  "sample_rate": 16000,
  "duration_seconds": 1.8,
  "cached": false
}
```

---

### Codes d'erreur

| Code | Signification |
|---|---|
| `400` | Langue non reconnue |
| `403` | Clé API invalide ou manquante |
| `422` | Corps invalide (champ manquant, valeur hors limites) |
| `429` | Rate limit atteint — réessaie dans 1 minute |
| `500` | Erreur interne du modèle |
| `504` | Timeout inférence — texte trop long |

---

## 📜 Historique

L'historique conserve le texte uniquement.
L'audio est régénéré à la demande via `/tts` — jamais stocké.

### `POST /history` → 201
```json
{
  "user_id": "user_001",
  "action_type": "translate",
  "src_lang": "francais",
  "tgt_lang": "moore",
  "source_text": "Bonjour",
  "result_text": "Ne y welame"
}
```

### `GET /history` → 200
```
GET /history?user_id=user_001&action_type=translate&lang=moore&limit=10&offset=0
```
Filtres : `user_id`, `action_type` (`translate`/`tts`/`translate_and_speak`), `lang`, `limit`, `offset`

### `DELETE /history/{id}` → 204 / 404

### `DELETE /history?user_id=xxx` → 204

### Migrations Alembic
```bash
alembic upgrade head
```

---

## 🧪 Tester avec Postman

1. Importe `FasoConnect_API.postman_collection.json`
2. Configure la variable `{{api_key}}` dans la collection
3. Lance **Health Check** en premier
4. Swagger UI : `http://localhost:8000/docs`

---

## 🔊 Écouter l'audio TTS

Ouvre `audio_player.html` dans ton navigateur.

| Onglet | Fonction |
|---|---|
| **Synthèse vocale** | Texte → audio direct |
| **Traduire + Écouter** | Pipeline complet en un clic |
| **Base64** | Colle un `audio_b64` reçu de Postman |
| **Historique** | Consulte, filtre et supprime l'historique |

---

## 🎓 Fine-tuning

### Pourquoi fine-tuner ?

NLLB-200 a été entraîné sur peu de données pour les langues burkinabè.
Le fine-tuning améliore significativement la qualité sur `mos_Latn`, `dyu_Latn` et `fuv_Latn`.

### Sources de données disponibles

| Dataset | Langues | Volume | Accès |
|---|---|---|---|
| Google SMOL (`smolsent` + `smoldoc`) | mos, dyu, ff | ~10 000–20 000 | ✅ Public |
| sawadogosalif/MooreFRCollections | mooré-français | variable | ✅ Public |
| OLDI Seed (successeur NLLB-Seed) | fuv_Latn | ~6 000 | ✅ Public |
| Bloom Library (sil-ai/bloom-lm) | mos, dyu, fuv | variable | ✅ Public |
| facebook/flores dev | mos, dyu, fuv | ~997×3 | ✅ (après CGU) |
| Archives RTB (radio) | mos, dyu, fuv | à collecter | 📧 Via demande |

> **Note :** JW300 est définitivement retiré d'OPUS depuis 2021 (droits d'auteur).

### Lancer le fine-tuning

```powershell
# Configuration du token HF
.\fix_hf_token.ps1

# Pipeline complet (sans transcription radio)
.\run_finetuning.ps1 -SkipRadio

# Ou étape par étape
python collect/download_all_datasets.py
python collect/clean_corpus.py
python training/evaluate_model.py --model facebook/nllb-200-distilled-600M
python training/finetune_nllb.py
python training/evaluate_model.py --compare
```

### Intégrer le modèle fine-tuné

Dans `translation/nllb_engine.py` :
```python
# Avant
MODEL_ID = "facebook/nllb-200-distilled-600M"

# Après fine-tuning
MODEL_ID = "./models/nllb-burkina-v1"
```

Puis rebuild :
```bash
docker compose up --build -d
```

---

## 📁 Structure du projet

```
fasoconnect/
├── api/
│   └── main.py                    API FastAPI (auth, rate limit, endpoints)
├── translation/
│   └── nllb_engine.py             Moteur NLLB-200
├── tts/
│   └── mms_engine.py              Moteur MMS-TTS
├── history/
│   ├── models.py                  Modèle SQLModel HistoryEntry
│   ├── database.py                Engine SQLite/PostgreSQL
│   └── router.py                  Endpoints /history
├── migrations/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/001_create_history_table.py
├── collect/
│   ├── download_all_datasets.py   Corpus publics (v6)
│   ├── clean_corpus.py            Nettoyage et déduplication
│   ├── transcribe_radio.py        Transcription Whisper (archives RTB)
│   └── download_opus_direct.py    Téléchargement OPUS direct (sans opustools)
├── training/
│   ├── config.py                  Hyperparamètres centralisés
│   ├── finetune_nllb.py           Pipeline fine-tuning
│   └── evaluate_model.py          BLEU/chrF/TER avant-après
├── tests/
│   └── test_history.py            25 tests pytest
├── data/
│   └── .gitkeep
├── preload_models.py
├── audio_player.html
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements_finetuning.txt
├── run_finetuning.ps1
├── fix_hf_token.ps1
├── .env
├── .env.example
├── .gitignore
└── README.md
```

---

## 📊 Performances (RTX 3060 Laptop — CUDA 13.3)

| Configuration | Chargement modèle | Traduction | TTS |
|---|---|---|---|
| CPU seul | 3–5 min | 10–30 sec | 5–15 sec |
| RTX 3060 (CUDA) | ~30 sec | < 1 sec | < 1 sec |

**Configuration WSL2 recommandée :**
```ini
[wsl2]
memory=10GB
processors=8
swap=4GB
gpuSupport=true
```

---

## 🔧 Dépannage

### Docker ne démarre pas
```powershell
wsl --shutdown
taskkill /f /im "Docker Desktop.exe"
# Relance Docker Desktop depuis le menu Démarrer
```

### `Child process died` au démarrage
RAM Docker insuffisante. Vérifie `~/.wslconfig` : `memory=10GB`

### `NllbTokenizer has no attribute lang_code_to_id`
Ancienne version de `transformers`. Le code utilise `convert_tokens_to_ids` — vérifie `nllb_engine.py`.

### `ModuleNotFoundError: No module named 'training'`
Lance les scripts depuis la racine du projet, pas depuis un sous-dossier :
```powershell
# Correct
cd C:\...\FasoConnect
python collect/clean_corpus.py

# Incorrect
cd collect
python clean_corpus.py
```

### `Redis connection refused`
L'hôte Redis doit être `redis` (nom du service Docker), pas `localhost`.

### GPU non détecté
```powershell
docker info | findstr nvidia
# Doit afficher : Runtimes: nvidia runc

# Si absent :
wsl -d Ubuntu
sudo nvidia-ctk runtime configure --runtime=docker
```

### `failed to create temp dir` au build Docker
Espace disque insuffisant :
```bash
docker system prune -a --volumes
```

### Rate limit atteint (429)
Attends 1 minute ou augmente les limites dans `api/main.py` :
```python
@limiter.limit("50/minute")  # augmenter si besoin
```

### Fine-tuning — corpus vide
```powershell
# Vérifie l'état du corpus
python -c "
import pandas as pd
df = pd.read_csv('data/processed/corpus_burkina_clean.csv')
print('Total :', len(df))
print(df.groupby('source').size())
"
```

---

## 📄 Licence

Projet académique — EILCO 2025/2026.
Modèles Meta NLLB-200 et MMS-TTS sous licence [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).