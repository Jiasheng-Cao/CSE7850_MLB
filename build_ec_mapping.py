import re
from pathlib import Path

DAT_PATH = Path("data/uniprot_sprot/uniprot_sprot.dat")

acc2ec = {}

current_accessions = []
current_ecs = []

with open(DAT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("ID"):
            current_accessions = []
            current_ecs = []

        elif line.startswith("AC"):
            accs = line[5:].strip().split(";")
            accs = [a.strip() for a in accs if a.strip()]
            current_accessions.extend(accs)

        elif line.startswith("DE") and "EC=" in line:
            matches = re.findall(r"EC=([\d\.-]+);", line)
            current_ecs.extend(matches)

        elif line.startswith("//"):
            if current_accessions and current_ecs:
                for acc in current_accessions:
                    acc2ec[acc] = current_ecs
            current_accessions = []
            current_ecs = []

print("Total proteins with EC:", len(acc2ec))

import pickle
with open("acc2ec.pkl", "wb") as f:
    pickle.dump(acc2ec, f)

print("Saved acc2ec.pkl")
