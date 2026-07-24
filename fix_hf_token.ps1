# fix_hf_token.ps1
# Configure le HF_TOKEN pour la session PowerShell courante
# et vérifie l'accès aux datasets nécessaires.
# Usage : .\fix_hf_token.ps1

param(
    [string]$Token = ""
)

function Write-Step($msg)  { Write-Host "`n$msg" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "  ✅ $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "  ❌ $msg" -ForegroundColor Red }
function Write-Info($msg)  { Write-Host "  ℹ️  $msg" -ForegroundColor Gray }

Write-Host ""
Write-Host "======================================================" -ForegroundColor Magenta
Write-Host "  FasoConnect — Fix HuggingFace Token"                  -ForegroundColor Magenta
Write-Host "======================================================" -ForegroundColor Magenta

# ── Étape 1 — Récupère le token ───────────────────────────────────────

Write-Step "1/4 — Vérification du HF_TOKEN..."

# Priorité : paramètre > .env > variable d'environnement
if ($Token -ne "") {
    $hfToken = $Token
    Write-Info "Token fourni en paramètre"
} elseif (Test-Path ".env") {
    $envContent = Get-Content ".env" | Where-Object { $_ -match "^HF_TOKEN=" }
    if ($envContent) {
        $hfToken = $envContent -replace "^HF_TOKEN=", ""
        Write-Info "Token lu depuis .env"
    }
} elseif ($env:HF_TOKEN) {
    $hfToken = $env:HF_TOKEN
    Write-Info "Token lu depuis variable d'environnement"
} else {
    Write-Err "HF_TOKEN introuvable"
    Write-Info "Solutions :"
    Write-Info "  1. Ajoute HF_TOKEN=hf_xxx dans ton fichier .env"
    Write-Info "  2. Ou passe le token en paramètre :"
    Write-Info "     .\fix_hf_token.ps1 -Token hf_xxxxxxxxxxxxx"
    Write-Info "  3. Crée un token sur : https://huggingface.co/settings/tokens"
    exit 1
}

if (-not $hfToken.StartsWith("hf_")) {
    Write-Warn "Le token ne commence pas par 'hf_' — vérifie sa validité"
}

Write-OK "HF_TOKEN trouvé : $($hfToken.Substring(0, [Math]::Min(8, $hfToken.Length)))..."

# ── Étape 2 — Injecte dans la session et dans .env ───────────────────

Write-Step "2/4 — Injection du token dans la session PowerShell..."

$env:HF_TOKEN = $hfToken
$env:HUGGING_FACE_HUB_TOKEN = $hfToken

Write-OK "Token injecté dans la session courante"
Write-Info "Note : valable uniquement pour cette session PowerShell"

# Vérifie que .env contient les deux variables
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw

    if ($envContent -notmatch "HUGGING_FACE_HUB_TOKEN=") {
        Add-Content ".env" "`nHUGGING_FACE_HUB_TOKEN=$hfToken"
        Write-OK "HUGGING_FACE_HUB_TOKEN ajouté dans .env"
    }
}

# ── Étape 3 — Login HuggingFace via Python ───────────────────────────

Write-Step "3/4 — Login HuggingFace Hub..."

$loginScript = @"
import os
from huggingface_hub import login, whoami
token = os.environ.get('HF_TOKEN', '')
if not token:
    print('ERROR: HF_TOKEN vide')
    exit(1)
try:
    login(token=token, add_to_git_credential=False)
    info = whoami()
    print(f'OK:{info[\"name\"]}')
except Exception as e:
    print(f'ERROR:{e}')
"@

$result = python -c $loginScript 2>&1
if ($result -like "OK:*") {
    $username = $result -replace "OK:", ""
    Write-OK "Connecté en tant que : $username"
} else {
    Write-Err "Login échoué : $result"
    Write-Info "Vérifie que ton token est valide sur :"
    Write-Info "https://huggingface.co/settings/tokens"
    exit 1
}

# ── Étape 4 — Vérification accès aux datasets ─────────────────────────

Write-Step "4/4 — Vérification accès aux datasets..."

$checkScript = @"
import os
os.environ['HF_TOKEN'] = '$hfToken'
from huggingface_hub import login
login(token='$hfToken', add_to_git_credential=False)

from datasets import load_dataset

checks = [
    ('facebook/flores',              'mos_Latn',         'FLORES-200 (public)'),
    ('google/smol',                  'smolsent__en_mos', 'Google SMOL mooré'),
    ('google/smol',                  'smolsent__en_dyu', 'Google SMOL dioula'),
    ('google/smol',                  'smolsent__en_ff',  'Google SMOL fulfulde'),
    ('Helsinki-NLP/bible-uedin',     'mos',              'Bible mooré'),
    ('Helsinki-NLP/opus-100',        'en-ff',            'OPUS-100 fulfulde'),
]

for dataset_id, config, label in checks:
    try:
        ds = load_dataset(dataset_id, config, split='train', streaming=True)
        row = next(iter(ds))
        print(f'OK:{label}')
    except Exception as e:
        err = str(e)[:80]
        print(f'FAIL:{label} — {err}')
"@

$results = python -c $checkScript 2>&1

foreach ($line in $results) {
    if ($line -like "OK:*") {
        $label = $line -replace "OK:", ""
        Write-OK $label
    } elseif ($line -like "FAIL:*") {
        $label = $line -replace "FAIL:", ""
        Write-Warn $label
    }
}

# ── Résumé ────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======================================================" -ForegroundColor Magenta
Write-Host "  Configuration terminée !" -ForegroundColor Green
Write-Host ""
Write-Host "  Lance maintenant le téléchargement des corpus :" -ForegroundColor White
Write-Host "  python collect/download_all_datasets.py" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Magenta