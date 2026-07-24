# run_finetuning.ps1 — VERSION FINALE
param(
    [switch]$SkipDownload,
    [switch]$SkipRadio,
    [switch]$SkipClean,
    [switch]$SkipEvalBase,
    [switch]$CompareOnly
)

# ── Ne pas bloquer sur les erreurs natives (stderr Python) ────────────
$ErrorActionPreference = "Continue"

function Write-Step($msg)  { Write-Host "`n$msg" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "  OK $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "  ERR $msg" -ForegroundColor Red }
function Write-Info($msg)  { Write-Host "  >> $msg" -ForegroundColor Gray }

Write-Host ""
Write-Host "======================================================" -ForegroundColor Magenta
Write-Host "  FasoConnect - Pipeline Fine-tuning NLLB-200"         -ForegroundColor Magenta
Write-Host "======================================================" -ForegroundColor Magenta

# ── Etape 1 - Verification environnement ─────────────────────────────

Write-Step "1/8 - Verification de l'environnement..."

$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Python non trouve"
    exit 1
}
Write-OK "Python : $pyVersion"

python -c "import torch; gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU uniquement'; print('  GPU :', gpu)" 2>$null

Write-Info "Installation des dependances..."
pip install -r requirements_finetuning.txt -q 2>$null
Write-OK "Dependances installees"

# ── Etape 2 - HuggingFace Token ──────────────────────────────────────

Write-Step "2/8 - Configuration HuggingFace Token..."

# Charge toutes les variables du .env dans la session
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]*)=(.*)$") {
            $key   = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-OK ".env charge"
} else {
    Write-Err "Fichier .env introuvable"
    exit 1
}

if (-not $env:HF_TOKEN) {
    Write-Err "HF_TOKEN manquant dans .env"
    exit 1
}

# Login HF via fichier Python temporaire (evite les problemes stderr PowerShell)
$loginScript = @"
import os, sys
os.environ['HF_TOKEN'] = r'$($env:HF_TOKEN)'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from huggingface_hub import login, whoami
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
info = whoami()
sys.stdout.write(info['name'])
"@

$tmpFile = [System.IO.Path]::GetTempFileName() + ".py"
$loginScript | Out-File -FilePath $tmpFile -Encoding utf8

$loginResult = python $tmpFile 2>$null
Remove-Item $tmpFile -ErrorAction SilentlyContinue

if ($loginResult) {
    Write-OK "HuggingFace connecte : $loginResult"
} else {
    Write-Warn "Login HF - verifie HF_TOKEN dans .env"
}

# ── Etape 3 - Creation des dossiers ──────────────────────────────────

Write-Step "3/8 - Creation des dossiers..."

$dirs = @(
    "data\raw\audio\moore",
    "data\raw\audio\dioula",
    "data\raw\audio\fulfulde",
    "data\raw\opus",
    "data\raw\transcriptions",
    "data\processed",
    "models",
    "logs\tensorboard",
    "logs\eval"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-OK "Dossiers crees"

# ── Etape 4 - Telechargement corpus ──────────────────────────────────

if (-not $SkipDownload -and -not $CompareOnly) {
    Write-Step "4/8 - Telechargement des corpus publics..."
    Write-Info "Sources : Google SMOL, JW300, Bible-uedin, FLORES, opus-100"

    python collect/download_all_datasets.py
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec telechargement corpus"
        exit 1
    }

    $corpusFile = "data\processed\corpus_burkina.csv"
    if (Test-Path $corpusFile) {
        $lineCount = (Get-Content $corpusFile | Measure-Object -Line).Lines
        Write-OK "Corpus telecharge : $lineCount lignes"
        if ($lineCount -lt 100) {
            Write-Warn "Corpus trop petit ($lineCount lignes) - verifie les logs"
        }
    } else {
        Write-Err "Fichier corpus non cree"
        exit 1
    }
} else {
    Write-Step "4/8 - Telechargement corpus (ignore)"
}

# ── Etape 5 - Transcription radio ────────────────────────────────────

if (-not $SkipRadio -and -not $CompareOnly) {
    Write-Step "5/8 - Transcription radio (Whisper)..."

    $hasAudio = $false
    foreach ($lang in @("moore", "dioula", "fulfulde")) {
        $langDir = "data\raw\audio\$lang"
        if ((Test-Path $langDir) -and (Get-ChildItem $langDir -File -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) {
            $hasAudio = $true
            break
        }
    }

    if ($hasAudio) {
        Write-Info "Fichiers audio detectes - lancement Whisper..."
        python collect/transcribe_radio.py
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Transcription terminee"
        } else {
            Write-Warn "Transcription echouee - etape ignoree"
        }
    } else {
        Write-Info "Aucun fichier audio - etape ignoree"
        Write-Info "Ajouter des fichiers dans data\raw\audio\moore\"
    }
} else {
    Write-Step "5/8 - Transcription radio (ignoree)"
}

# ── Etape 6 - Nettoyage corpus ────────────────────────────────────────

if (-not $SkipClean -and -not $CompareOnly) {
    Write-Step "6/8 - Nettoyage et deduplication..."

    $corpusFile = "data\processed\corpus_burkina.csv"
    if (-not (Test-Path $corpusFile)) {
        Write-Err "Corpus introuvable : $corpusFile"
        exit 1
    }

    python collect/clean_corpus.py
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec nettoyage corpus"
        exit 1
    }

    $cleanFile = "data\processed\corpus_burkina_clean.csv"
    if (Test-Path $cleanFile) {
        $lineCount = (Get-Content $cleanFile | Measure-Object -Line).Lines
        Write-OK "Corpus nettoye : $lineCount paires"
    }
} else {
    Write-Step "6/8 - Nettoyage corpus (ignore)"
}

# ── Etape 7 - Evaluation modele de base ──────────────────────────────

if (-not $SkipEvalBase -and -not $CompareOnly) {
    Write-Step "7/8 - Evaluation du modele de BASE..."

    python training/evaluate_model.py --model facebook/nllb-200-distilled-600M
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Scores de base sauvegardes dans logs\eval\"
    } else {
        Write-Warn "Evaluation base echouee - on continue"
    }
} else {
    Write-Step "7/8 - Evaluation modele de base (ignoree)"
}

# ── Etape 8 - Fine-tuning ─────────────────────────────────────────────

if (-not $CompareOnly) {
    Write-Step "8/8 - Fine-tuning NLLB-200..."
    Write-Info "Duree estimee : 2-8h (RTX 3060)"
    Write-Info "Suivi en temps reel dans un autre terminal :"
    Write-Info "  tensorboard --logdir logs\tensorboard"

    $cleanCorpus = "data\processed\corpus_burkina_clean.csv"
    if (-not (Test-Path $cleanCorpus)) {
        Write-Err "Corpus nettoye introuvable : $cleanCorpus"
        exit 1
    }

    python training/finetune_nllb.py
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Fine-tuning echoue"
        exit 1
    }
    Write-OK "Fine-tuning termine -> models\nllb-burkina-v1\"
}

# ── Comparaison finale ────────────────────────────────────────────────

Write-Step "Comparaison base vs fine-tune..."

if (Test-Path "models\nllb-burkina-v1") {
    python training/evaluate_model.py --compare
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Resultats dans logs\eval\comparison_results.json"
    }
} else {
    Write-Warn "Modele fine-tune introuvable - comparaison ignoree"
}

# ── Resume ────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======================================================" -ForegroundColor Magenta
Write-Host "  Pipeline termine !" -ForegroundColor Green
Write-Host ""
Write-Host "  Integrer dans FasoConnect :" -ForegroundColor White
Write-Host "  1. translation\nllb_engine.py :" -ForegroundColor White
Write-Host "     MODEL_ID = './models/nllb-burkina-v1'" -ForegroundColor Yellow
Write-Host "  2. docker compose up --build -d" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Magenta