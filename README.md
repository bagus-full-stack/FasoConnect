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
- [Tester avec Postman](#-tester-avec-postman)
- [Écouter l'audio TTS](#-écouter-laudio-tts)
- [Structure du projet](#-structure-du-projet)
- [Performances](#-performances)
- [Dépannage](#-dépannage)

---

## 🗣️ Langues supportées

| Langue | Code NLLB | Traduction | TTS |
|---|---|---|---|
| Mooré | `mos_Latn` | ✅ | ✅ |
| Dioula / Jula | `dyu_Latn` | ✅ | ✅ |
| Fulfulde | `fuv_Latn` | ✅ | ✅ |
| Gourmantchéma | `gux_Latn` | ✅ | ✅ |
| Dagaare | `dga_Latn` | ✅ | ✅ |
| Français | `fra_Latn` | ✅ | ✅ |
| Anglais | `eng_Latn` | ✅ | ✅ |

Toutes les combinaisons bidirectionnelles sont supportées :
`Français ↔ Langue locale`, `Anglais ↔ Langue locale`, `Langue locale ↔ Langue locale`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Client (Postman / Angular)          │
└──────────────────────────┬──────────────────────────┘
                           │ HTTP + X-API-Key
┌──────────────────────────▼──────────────────────────┐
│              FastAPI (Uvicorn, port 8000)            │
│  ┌────────────────┐   ┌─────────────────────────┐   │
│  │  NLLB-200      │   │  MMS-TTS                │   │
│  │  (Traduction)  │   │  (Synthèse vocale)      │   │
│  │  ~2.6 Go VRAM  │   │  ~1 Go VRAM             │   │
│  └────────────────┘   └─────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐     │
│  │  Redis Cache (TTL 1h)                       │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
         │
         └── RTX 3060 (CUDA) / CPU fallback
```

**Stack technique :**
- `FastAPI` — framework API async
- `Meta NLLB-200 1.3B` — modèle de traduction 200 langues
- `Meta MMS-TTS` — synthèse vocale multilingue
- `Redis` — cache des traductions et audio
- `Docker + Docker Compose` — containerisation
- `SlowAPI` — rate limiting
- `HuggingFace Transformers` — chargement des modèles

---

## ⚙️ Prérequis

| Outil | Version minimale |
|---|---|
| Docker Desktop | 4.x+ |
| WSL2 (Windows) | Ubuntu 22.04 |
| GPU NVIDIA (optionnel) | Driver 610+ |
| RAM | 10 Go alloués à WSL |
| Disque | 15 Go libres |

> **Sans GPU :** l'API fonctionne en CPU mais les inférences prennent 10 à 30 secondes.  
> **Avec GPU RTX 3060 :** moins d'1 seconde par traduction.

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/ton-user/fasoconnect.git
cd fasoconnect
```

### 2. Configurer l'environnement

```bash
cp .env.example .env
```

Édite `.env` avec tes valeurs :

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
API_KEY=ta_cle_generee_ici
ALLOWED_ORIGIN=http://localhost:4200
```

**Générer une API Key forte :**
```bash
# Dans WSL
openssl rand -hex 32
```

### 3. Configurer WSL2 (Windows uniquement)

Crée ou édite `C:\Users\<ton-user>\.wslconfig` :

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

### 4. Activer le GPU dans Docker (optionnel)

```bash
# Dans Ubuntu WSL
sudo nvidia-ctk runtime configure --runtime=docker
```

Redémarre Docker Desktop, puis vérifie :
```bash
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

---

## ⚙️ Configuration

Toute la configuration se fait via le fichier `.env` :

| Variable | Description | Défaut |
|---|---|---|
| `HF_TOKEN` | Token HuggingFace ([créer](https://huggingface.co/settings/tokens)) | — |
| `API_KEY` | Clé d'accès à l'API | — |
| `ALLOWED_ORIGIN` | Domaine autorisé pour les requêtes CORS | `*` |
| `CACHE_TTL` | Durée du cache Redis en secondes | `3600` |
| `INFER_TIMEOUT` | Timeout des inférences en secondes | `60` |

---

## ▶️ Lancer l'API

### Premier lancement (~10 minutes)
```bash
docker compose up --build -d
```
> Le premier lancement télécharge les modèles (~5 Go). Les suivants démarrent en 30 secondes.

### Lancer sans rebuild
```bash
docker compose up -d
```

### Voir les logs
```bash
docker compose logs -f api
```

### Arrêter
```bash
docker compose down
```

### Logs attendus au démarrage
```
✅ HuggingFace authentifié
⏳ Chargement NLLB-200...
✅ NLLB-200 chargé sur cuda
⏳ Chargement MMS-TTS...
✅ MMS-TTS prêt
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 📡 Endpoints

### Base URL
```
http://localhost:8000
```

### Authentification
Ajouter le header suivant à chaque requête :
```
X-API-Key: ta_cle_api
```

---

### `GET /health`
Vérifie que l'API est opérationnelle.

**Réponse :**
```json
{
  "status": "ok",
  "models": ["nllb-200", "mms-tts"],
  "cache": "redis",
  "version": "1.0.0"
}
```

---

### `GET /languages`
Liste toutes les langues supportées.

**Réponse :**
```json
{
  "translation_supported": ["moore", "dioula", "fulfulde", "gourmantsema", "dagaare", "francais", "anglais"],
  "tts_supported": ["moore", "dioula", "fulfulde", "gourmantsema", "dagaare", "francais", "anglais"],
  "nllb_codes": { "moore": "mos_Latn", "..." : "..." }
}
```

---

### `POST /translate`
Traduit un texte entre deux langues.

**Rate limit :** 20 requêtes/minute par IP

**Body :**
```json
{
  "text": "Bonjour, comment allez-vous ?",
  "src_lang": "francais",
  "tgt_lang": "moore"
}
```

**Réponse :**
```json
{
  "translated_text": "Ne y welame ?",
  "src_lang": "francais",
  "tgt_lang": "moore",
  "cached": false
}
```

---

### `POST /tts`
Génère un audio WAV encodé en base64.

**Rate limit :** 10 requêtes/minute par IP

**Body :**
```json
{
  "text": "Ne y welame",
  "lang": "moore",
  "speed": 1.0
}
```

**Réponse :**
```json
{
  "audio_b64": "UklGRiQ...",
  "sample_rate": 16000,
  "duration_seconds": 2.3,
  "lang": "moore"
}
```

**Paramètre `speed` :** `0.5` (lent) → `1.0` (normal) → `2.0` (rapide)

---

### `POST /translate-and-speak`
Pipeline combiné : traduction + synthèse vocale en une requête.

**Rate limit :** 10 requêtes/minute par IP

**Body :**
```json
{
  "text": "Bonjour tout le monde",
  "src_lang": "francais",
  "tgt_lang": "dioula",
  "speed": 1.0
}
```

**Réponse :**
```json
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
| `422` | Corps de requête invalide (champ manquant, valeur hors limites) |
| `429` | Trop de requêtes — rate limit atteint |
| `500` | Erreur interne du modèle |
| `504` | Timeout — inférence trop longue |

---

## 🧪 Tester avec Postman

1. Importe `FasoConnect_API.postman_collection.json` dans Postman
2. La variable `{{base_url}}` est préconfigurée sur `http://localhost:8000`
3. Ajoute le header `X-API-Key` dans chaque requête ou au niveau de la collection
4. Lance d'abord **Health Check** pour vérifier que l'API répond

La **Swagger UI** est aussi disponible à :
```
http://localhost:8000/docs
```

---

## 🔊 Écouter l'audio TTS

Ouvre `audio_player.html` dans ton navigateur — il se connecte directement à `localhost:8000` et permet de :
- Synthétiser un texte et l'écouter directement
- Traduire puis écouter en une seule action
- Coller un `audio_b64` reçu depuis Postman

---

## 📁 Structure du projet

```
fasoconnect/
├── api/
│   └── main.py               # API FastAPI (endpoints, auth, rate limiting)
├── translation/
│   └── nllb_engine.py        # Moteur de traduction NLLB-200
├── tts/
│   └── mms_engine.py         # Moteur de synthèse vocale MMS-TTS
├── preload_models.py          # Script de téléchargement des modèles au build
├── audio_player.html          # Lecteur audio pour le TTS
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                       # Secrets (non commité)
├── .env.example               # Template de configuration
├── .gitignore
└── README.md
```

---

## 📊 Performances

| Configuration | Chargement modèle | Traduction | TTS |
|---|---|---|---|
| CPU seul | 3–5 min | 10–30 sec | 5–15 sec |
| RTX 3060 (CUDA) | ~30 sec | < 1 sec | < 1 sec |

**Configuration recommandée (testée) :**
- `NLLB-200 distilled 1.3B` en `float16` sur GPU
- WSL2 avec 10 Go RAM alloués
- 1 worker Uvicorn (modèle trop lourd pour plusieurs workers)

---

## 🔧 Dépannage

### Docker ne démarre pas
```powershell
wsl --shutdown
taskkill /f /im "Docker Desktop.exe"
# Relance Docker Desktop depuis le menu Démarrer
```

### `Child process died` au démarrage
RAM Docker insuffisante. Vérifie `~/.wslconfig` :
```ini
[wsl2]
memory=10GB
```

### `NllbTokenizer has no attribute lang_code_to_id`
Erreur de version de `transformers`. Le code utilise déjà `convert_tokens_to_ids` — vérifie que tu as bien la dernière version de `nllb_engine.py`.

### `Redis connection refused`
L'adresse Redis doit être `redis` (nom du service Docker) et non `localhost`.

### GPU non détecté
```bash
# Vérifie le runtime NVIDIA
docker info | findstr nvidia
# Doit afficher : Runtimes: nvidia runc

# Reconfigure si absent
wsl -d Ubuntu
sudo nvidia-ctk runtime configure --runtime=docker
```

### `failed to create temp dir` au build
Espace disque Docker insuffisant :
```bash
docker system prune -a --volumes
```

---

## 📄 Licence

Projet académique — EILCO 2025/2026.  
Modèles Meta NLLB-200 et MMS-TTS sous licence [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).