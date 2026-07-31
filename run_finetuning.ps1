# run_finetuning.ps1 — VERSION FINALE
#
# Usage :
#   .\run_finetuning.ps1                   <- pipeline complet
#   .\run_finetuning.ps1 -SkipDownload     <- ignore le telechargement
#   .\run_finetuning.ps1 -SkipRadio        <- ignore la transcription radio
#   .\run_finetuning.ps1 -SkipClean        <- ignore le nettoyage
#   .\run_finetuning.ps1 -SkipEvalBase     <- ignore evaluation de base
#   .\run_finetuning.ps1 -CompareOnly      <- uniquement la comparaison finale

param(
    [switch]$SkipDownload,
    [switch]$SkipRadio,
    [switch]$SkipClean,
    [switch]$SkipEvalBase,
    [switch]$CompareOnly
)

# Ne pas bloquer sur stderr Python
$ErrorActionPreference = "Continue"

function Write-Step($msg)  { Write-Host "`n$msg" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "  OK $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "  ERR $msg" -ForegroundColor Red }
function Write-Info($msg)  { Write-Host "  >> $msg" -ForegroundColor Gray }

function Load-DotEnv {
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^([^#][^=]*)=(.*)$") {
                [System.Environment]::SetEnvironmentVariable(
                    $Matches[1].Trim(), $Matches[2].Trim(), "Process"
                )
            }
        }
        return $true
    }
    return $false
}

function HF-Login {
    $token = $env:HF_TOKEN
    if (-not $token) { return $false }

    $script = @"
import os, sys, warnings
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
os.environ['HF_TOKEN'] = r'$token'
from huggingface_hub import login, whoami
login(token=r'$token', add_to_git_credential=False)
info = whoami()
sys.stdout.write(info['name'])
"@
    $tmp = [System.IO.Path]::GetTempFileName() + ".py"
    $script | Out-File -FilePath $tmp -Encoding utf8
    $result = python $tmp 2>$null
    Remove-Item $tmp -ErrorAction SilentlyContinue
    return $result
}

# ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======================================================" -ForegroundColor Magenta
Write-Host "  FasoConnect - Fine-tuning NLLB-200 (Pipeline Final)" -ForegroundColor Magenta
Write-Host "======================================================" -ForegroundColor Magenta

# ── 1/8 Environnement ────────────────────────────────────────────────

Write-Step "1/8 - Verification de l'environnement..."

$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Python non trouve. Installe Python 3.10+"
    exit 1
}
Write-OK "Python : $pyVer"

python -c "
import torch
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print('  GPU :', dev)
" 2>$null

Write-Info "Installation des dependances..."
pip install -r requirements_finetuning.txt -q 2>$null
Write-OK "Dependances OK"

# ── 2/8 HuggingFace Token ────────────────────────────────────────────

Write-Step "2/8 - Configuration HuggingFace..."

if (-not (Load-DotEnv)) {
    Write-Err ".env introuvable - copie .env.example en .env"
    exit 1
}
Write-OK ".env charge"

if (-not $env:HF_TOKEN) {
    Write-Err "HF_TOKEN manquant dans .env"
    exit 1
}

$username = HF-Login
if ($username) {
    Write-OK "HuggingFace connecte : $username"
} else {
    Write-Warn "Login HF echoue - verifie HF_TOKEN"
}

# ── 3/8 Dossiers ─────────────────────────────────────────────────────

Write-Step "3/8 - Creation des dossiers..."

@(
    "data\raw\audio\moore",
    "data\raw\audio\dioula",
    "data\raw\audio\fulfulde",
    "data\raw\opus",
    "data\raw\transcriptions",
    "data\processed",
    "models",
    "logs\tensorboard",
    "logs\eval"
) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}
Write-OK "Dossiers crees"

# ── 4/8 Telechargement corpus ─────────────────────────────────────────

if (-not $SkipDownload -and -not $CompareOnly) {
    Write-Step "4/8 - Telechargement des corpus..."
    Write-Info "Sources : Google SMOL, MooreFRCollections, OLDI Seed, Bloom, FLORES"
    Write-Info "Duree estimee : 5-20 min"

    python collect/download_all_datasets.py
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec telechargement"
        exit 1
    }

    $f = "data\processed\corpus_burkina.csv"
    if (Test-Path $f) {
        $n = (Get-Content $f | Measure-Object -Line).Lines
        Write-OK "Corpus telecharge : $n lignes"
        if ($n -lt 100) {
            Write-Warn "Corpus trop petit ($n lignes) - verifie les logs"
        }
    } else {
        Write-Err "Fichier corpus non cree"
        exit 1
    }
} else {
    Write-Step "4/8 - Telechargement (ignore)"
}

# ── 5/8 Transcription radio ───────────────────────────────────────────

if (-not $SkipRadio -and -not $CompareOnly) {
    Write-Step "5/8 - Transcription radio (Whisper)..."

    $hasAudio = $false
    foreach ($lang in @("moore", "dioula", "fulfulde")) {
        $d = "data\raw\audio\$lang"
        if ((Test-Path $d) -and (Get-ChildItem $d -File -EA SilentlyContinue | Measure-Object).Count -gt 0) {
            $hasAudio = $true; break
        }
    }

    if ($hasAudio) {
        Write-Info "Fichiers audio detectes - lancement Whisper..."
        python collect/transcribe_radio.py
        if ($LASTEXITCODE -eq 0) { Write-OK "Transcription terminee" }
        else { Write-Warn "Transcription echouee - etape ignoree" }
    } else {
        Write-Info "Aucun fichier audio - etape ignoree"
        Write-Info "Ajouter des fichiers dans data\raw\audio\moore\"
    }
} else {
    Write-Step "5/8 - Transcription radio (ignoree)"
}

# ── 6/8 Nettoyage corpus ──────────────────────────────────────────────

if (-not $SkipClean -and -not $CompareOnly) {
    Write-Step "6/8 - Nettoyage et deduplication..."

    if (-not (Test-Path "data\processed\corpus_burkina.csv")) {
        Write-Err "Corpus brut introuvable - lance d'abord l'etape 4"
        exit 1
    }

    python collect/clean_corpus.py
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Echec nettoyage"
        exit 1
    }

    $f = "data\processed\corpus_burkina_clean.csv"
    if (Test-Path $f) {
        $n = (Get-Content $f | Measure-Object -Line).Lines
        Write-OK "Corpus nettoye : $n paires -> $f"

        # Affiche la repartition
        python -c "
import pandas as pd
df = pd.read_csv('data/processed/corpus_burkina_clean.csv')
train = df[df.get('source','') != 'flores_dev'] if 'source' in df.columns else df
print(f'  Entrainement : {len(train):,} paires')
print(f'  Validation   : {len(df)-len(train):,} paires')
if 'source' in df.columns and len(train) > 0:
    print('  Par source :')
    for src, cnt in train.groupby('source').size().items():
        print(f'    {src}: {cnt:,}')
" 2>$null
    }
} else {
    Write-Step "6/8 - Nettoyage (ignore)"
}

# ── 7/8 Evaluation modele de base ─────────────────────────────────────

if (-not $SkipEvalBase -and -not $CompareOnly) {
    Write-Step "7/8 - Evaluation du modele de BASE (reference)..."
    Write-Info "Score BLEU avant fine-tuning sur FLORES-200"

    python -m training.evaluate_model --model facebook/nllb-200-distilled-600M
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Scores de base sauvegardes dans logs\eval\"
    } else {
        Write-Warn "Evaluation base echouee - on continue"
    }
} else {
    Write-Step "7/8 - Evaluation modele de base (ignoree)"
}

# ── 8/8 Fine-tuning ───────────────────────────────────────────────────

if (-not $CompareOnly) {
    Write-Step "8/8 - Fine-tuning NLLB-200..."
    Write-Info "Duree estimee : 2-8h sur RTX 3060"
    Write-Info "Suivi en temps reel (autre terminal) :"
    Write-Info "  tensorboard --logdir logs\tensorboard"
    Write-Info "  Puis ouvre http://localhost:6006"
    Write-Host ""

    $cleanCorpus = "data\processed\corpus_burkina_clean.csv"
    if (-not (Test-Path $cleanCorpus)) {
        Write-Err "Corpus nettoye introuvable : $cleanCorpus"
        Write-Info "Lance d'abord l'etape 6 (nettoyage)"
        exit 1
    }

    # Verifie qu'il y a assez de donnees
    $lines = (Get-Content $cleanCorpus | Measure-Object -Line).Lines
    if ($lines -lt 10) {
        Write-Err "Corpus trop petit ($lines lignes) pour le fine-tuning"
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
    } else {
        Write-Warn "Comparaison echouee"
    }
} else {
    Write-Warn "Modele fine-tune introuvable - comparaison ignoree"
}

# ── Resume ────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======================================================" -ForegroundColor Magenta
Write-Host "  Pipeline termine !" -ForegroundColor Green
Write-Host ""
Write-Host "  Pour integrer le modele fine-tune :" -ForegroundColor White
Write-Host "  1. Modifie translation\nllb_engine.py :" -ForegroundColor White
Write-Host "     MODEL_ID = './models/nllb-burkina-v1'" -ForegroundColor Yellow
Write-Host "  2. Rebuild l'API :" -ForegroundColor White
Write-Host "     docker compose up --build -d" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Magenta