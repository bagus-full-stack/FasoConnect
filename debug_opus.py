# debug_opus.py — Diagnostique le problème opustools
# Usage : python debug_opus.py

import sys
import subprocess

print("=== Test 1 : opustools installé ? ===")
try:
    import opustools
    print(f"✅ opustools installé : {opustools.__version__}")
except ImportError as e:
    print(f"❌ opustools non installé : {e}")
    print("   Fix : pip install opustools")
    sys.exit(1)

print("\n=== Test 2 : commande opus_read disponible ? ===")
result = subprocess.run(
    [sys.executable, "-m", "opustools.opus_read", "--help"],
    capture_output=True, text=True, timeout=10
)
if result.returncode == 0:
    print("✅ opus_read fonctionne")
    print(result.stdout[:200])
else:
    print(f"❌ opus_read échoue (code {result.returncode})")
    print("stdout:", result.stdout[:200])
    print("stderr:", result.stderr[:200])

print("\n=== Test 3 : connexion OPUS server ===")
try:
    import urllib.request
    req = urllib.request.urlopen("https://opus.nlpl.eu/", timeout=10)
    print(f"✅ OPUS accessible : HTTP {req.status}")
except Exception as e:
    print(f"❌ OPUS inaccessible : {e}")

print("\n=== Test 4 : liste des langues JW300 disponibles ===")
result2 = subprocess.run(
    [sys.executable, "-m", "opustools.opus_read",
     "-d", "JW300", "-s", "mos", "-t", "fr",
     "--list_resources"],
    capture_output=True, text=True, timeout=30
)
print("stdout:", result2.stdout[:500])
print("stderr:", result2.stderr[:500])