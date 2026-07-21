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

| Langue | Code NLLB | Code interne | Traduction | TTS |
|---|---|---|---|---|
| Mooré | `mos_Latn` | `moore` | ✅ | ✅ |
| Dioula / Jula | `dyu_Latn` | `dioula` | ✅ | ✅ |
| Fulfulde | `fuv_Latn` | `fulfulde` | ✅ | ✅ |
| Gourmantchéma | `gux_Latn` | `gourmantsema` | ✅ | ✅ |
| Dagaare | `dga_Latn` | `dagaare` | ✅ | ✅ |
| Français | `fra_Latn` | `francais` | ✅ | ✅ |
| Anglais | `eng_Latn` | `anglais` | ✅ | ✅ |

Toutes les combinaisons bidirectionnelles sont supportées.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│         Clients (Postman · audio_player · Angular)  │
└──────────────────────────┬──────────────────────────┘
                           │ HTTPS + X-API-Key
┌──────────────────────────▼──────────────────────────┐
│    FastAPI — Auth · Rate Limit · CORS · Logs JSON   │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ /translate  │  │  /tts    │  │ /history       │  │
│  │ /translate- │  │          │  │ GET POST DELETE│  │
│  │ and-speak   │  │          │  │                │  │
│  └──────┬──────┘  └────┬─────┘  └───────┬────────┘  │
│         │              │                │            │
│  ┌──────▼──────┐  ┌────▼─────┐  ┌──────▼──────────┐ │
│  │ NLLB-200   │  │ MMS-TTS  │  │ SQLite/PostgreSQL│ │
│  │ 600M CUDA  │  │ CUDA     │  │ + Alembic       │ │
│  └─────────────┘  └──────────┘  └─────────────────┘ │
│  ┌─────────────────────────────────────────────────┐ │
│  │            Redis Cache (TTL 1h)                 │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────── Docker Compose ─────────────┘
                    WSL2 · RTX 3060 · CUDA
```

---

## ⚙️ Prérequis

| Outil | Version |
|---|---|
| Docker Desktop | 4.x+ |
| WSL2 + Ubuntu 22.04 | — |
| GPU NVIDIA (optionnel) | Driver 610+ |
| RAM allouée à WSL | 10 Go minimum |
| Espace disque libre | 15 Go minimum |

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

Crée `C:\Users\<user>\.wslconfig` :

```ini
[wsl2]
memory=10GB
processors=8
swap=4GB
gpuSupport=true
networkingMode=mirrored
```

```powershell
wsl --shutdown
```

### 5 — Activer le GPU dans Docker

```bash
wsl -d Ubuntu
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
| `HF_TOKEN` | Token HuggingFace | — |
| `API_KEY` | Clé d'accès à l'API | — |
| `ALLOWED_ORIGIN` | Domaine CORS autorisé | `*` |
| `DATABASE_URL` | URL base de données | `sqlite:///./fasoconnect.db` |
| `CACHE_TTL` | Durée cache Redis (secondes) | `3600` |
| `INFER_TIMEOUT` | Timeout inférence (secondes) | `60` |

---

## ▶️ Lancer l'API

```bash
# Premier lancement (~10 min — télécharge les modèles)
docker compose up --build -d

# Lancements suivants (~30 sec)
docker compose up -d

# Logs
docker compose logs -f api

# Arrêt
docker compose down
```

Logs attendus :
```
✅ HuggingFace authentifié
✅ Base de données prête
✅ NLLB-200 chargé sur cuda
✅ MMS-TTS prêt
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 📡 Endpoints

Base URL : `http://localhost:8000`

Auth : header `X-API-Key: ta_cle` sur tous les endpoints (si `API_KEY` configurée).

Swagger UI : `http://localhost:8000/docs`

---

### GET `/health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models": ["nllb-200-distilled-600M", "mms-tts"],
  "cache": "redis"
}
```

---

### GET `/languages`

```json
{
  "translation_supported": ["moore", "dioula", "fulfulde", "gourmantsema", "dagaare", "francais", "anglais"],
  "tts_supported": ["moore", "dioula", "fulfulde", "gourmantsema", "dagaare", "francais", "anglais"],
  "nllb_codes": { "moore": "mos_Latn", "...": "..." }
}
```

---

### POST `/translate`

Rate limit : 20 req/min par IP.

```json
// Body
{
  "text": "Bonjour, comment allez-vous ?",
  "src_lang": "francais",
  "tgt_lang": "moore"
}

// Réponse 200
{
  "translated_text": "Ne y welame ?",
  "src_lang": "francais",
  "tgt_lang": "moore",
  "cached": false
}
```

---

### POST `/tts`

Rate limit : 10 req/min par IP.

```json
// Body
{
  "text": "Ne y welame",
  "lang": "moore",
  "speed": 1.0
}

// Réponse 200
{
  "audio_b64": "UklGRiQ...",
  "sample_rate": 16000,
  "duration_seconds": 2.3,
  "lang": "moore"
}
```

`speed` : `0.5` (lent) → `1.0` (normal) → `2.0` (rapide).

---

### POST `/translate-and-speak`

Rate limit : 10 req/min par IP.

```json
// Body
{
  "text": "Bonjour tout le monde",
  "src_lang": "francais",
  "tgt_lang": "dioula",
  "speed": 1.0
}

// Réponse 200
{
  "original_text": "Bonjour tout le monde",
  "translated_text": "I ni ce",
  "src_lang": "francais",
  "tgt_lang": "dioula",
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
| `429` | Rate limit atteint |
| `500` | Erreur interne du modèle |
| `504` | Timeout inférence |

---

## 📜 Historique

L'historique conserve le texte uniquement — l'audio est régénéré à la demande via `/tts`.

### POST `/history` — Créer une entrée (201)

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

### GET `/history` — Liste paginée (200)

```
GET /history?user_id=user_001&action_type=translate&lang=moore&limit=10&offset=0
```

```json
{
  "total": 42,
  "limit": 10,
  "offset": 0,
  "items": [...]
}
```

Filtres disponibles : `user_id`, `action_type`, `lang`, `limit`, `offset`.

### DELETE `/history/{id}` — Supprimer une entrée (204 / 404)

### DELETE `/history?user_id=xxx` — Vider l'historique d'un utilisateur (204)

### Migrations Alembic

```bash
alembic upgrade head
```

---

## 🧪 Tester avec Postman

1. Importe `FasoConnect_API.postman_collection.json`
2. Configure la variable `api_key` dans la collection
3. Lance d'abord **Health Check**
4. La Swagger UI est disponible sur `http://localhost:8000/docs`

---

## 🔊 Écouter l'audio TTS

Ouvre `audio_player.html` dans ton navigateur.

Fonctionnalités :
- Onglet **Synthèse** : synthétise un texte directement
- Onglet **Traduire + Écouter** : pipeline complet en un clic
- Onglet **Base64** : colle un `audio_b64` reçu depuis Postman
- Onglet **Historique** : consulte, filtre et supprime l'historique

---

## 🎓 Fine-tuning

Pour améliorer la qualité sur les langues locales, un pipeline de fine-tuning est disponible dans `training/`.

### Données d'entraînement recommandées

| Dataset | Usage | Volume |
|---|---|---|
| JW300 (OPUS) | Entraînement | ~100k phrases |
| allenai/nllb | Entraînement | ~50k phrases |
| Google SMOL | Entraînement | ~5k phrases |
| NLLB-SEED | Entraînement | ~6k phrases |
| FLORES+ dev | Validation | ~1k phrases |
| FLORES-200 devtest | Évaluation finale | ~1k phrases |

### Lancer le fine-tuning

```bash
# Télécharger les corpus
python collect/download_all_datasets.py

# Transcrire les archives radio (optionnel)
python collect/transcribe_radio.py

# Lancer l'entraînement
python training/finetune_nllb.py
```

Après fine-tuning, mettre à jour `MODEL_ID` dans `nllb_engine.py` :

```python
MODEL_ID = "./models/nllb-burkina-v1"
```

---

## 📁 Structure du projet

```
fasoconnect/
├── api/
│   └── main.py                  ← API principale (auth, rate limit, endpoints)
├── translation/
│   └── nllb_engine.py           ← Moteur NLLB-200
├── tts/
│   └── mms_engine.py            ← Moteur MMS-TTS
├── history/
│   ├── models.py                ← Modèle SQLModel HistoryEntry
│   ├── database.py              ← Engine SQLite/PostgreSQL
│   └── router.py                ← Endpoints /history
├── migrations/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│       └── 001_create_history_table.py
├── collect/
│   ├── download_all_datasets.py ← Téléchargement corpus
│   └── transcribe_radio.py      ← Transcription Whisper
├── training/
│   └── finetune_nllb.py         ← Pipeline fine-tuning
├── tests/
│   └── test_history.py          ← 25 tests (pytest)
├── models/                      ← Modèles fine-tunés (non commités)
├── data/                        ← Corpus (non commité)
├── preload_models.py            ← Téléchargement au build Docker
├── audio_player.html            ← Lecteur audio (4 onglets)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                         ← Secrets (non commité)
├── .env.example
├── .gitignore
└── README.md
```

---

## 📊 Performances

| Config | Chargement | Traduction | TTS |
|---|---|---|---|
| CPU seul | 3–5 min | 10–30 sec | 5–15 sec |
| RTX 3060 CUDA | ~30 sec | < 1 sec | < 1 sec |

Config testée : NLLB-200 600M · float16 · WSL2 10 Go RAM · RTX 3060 6 Go VRAM.

---

## 🔧 Dépannage

**Docker ne démarre pas**
```powershell
wsl --shutdown
taskkill /f /im "Docker Desktop.exe"
# Relance Docker Desktop
```

**`Child process died` au démarrage**
RAM Docker insuffisante. Vérifie `~/.wslconfig` : `memory=10GB`.

**`NllbTokenizer has no attribute lang_code_to_id`**
Ancienne version de `transformers`. Le code utilise `convert_tokens_to_ids` — vérifie `nllb_engine.py`.

**`Redis connection refused`**
L'hôte Redis doit être `redis` (nom du service Docker), pas `localhost`.

**GPU non détecté**
```bash
docker info | findstr nvidia
# Doit afficher : Runtimes: nvidia runc
sudo nvidia-ctk runtime configure --runtime=docker
```

**`failed to create temp dir` au build**
Espace disque insuffisant : `docker system prune -a --volumes`.

**Rate limit atteint (429)**
Attends 1 minute ou augmente les limites dans `main.py` (endpoints `@limiter.limit`).

---

## 📄 Licence

Projet académique — EILCO 2025/2026.
Modèles Meta NLLB-200 et MMS-TTS sous licence [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).