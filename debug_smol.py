# debug_smol.py — Lance ce script pour voir la structure exacte de SMOL
# Usage : python debug_smol.py

from datasets import load_dataset

print("Chargement smolsent__en_mos en streaming...")
ds = load_dataset("google/smol", "smolsent__en_mos", split="train", streaming=True)

print("\n=== Structure de la première ligne ===")
row = next(iter(ds))
for key, value in row.items():
    print(f"  {key!r}: {str(value)[:100]!r}")

print("\n=== 3 premières lignes complètes ===")
ds2 = load_dataset("google/smol", "smolsent__en_mos", split="train", streaming=True)
for i, row in enumerate(ds2):
    if i >= 3:
        break
    print(f"\nLigne {i+1} :")
    for k, v in row.items():
        print(f"  {k}: {str(v)[:120]}")