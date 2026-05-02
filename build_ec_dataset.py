import pickle
import torch
from pathlib import Path

DATA_PKL = Path("data/swissprot_exp_2023_03.pkl")
ACC2EC_PKL = Path("acc2ec.pkl")

df = pickle.load(open(DATA_PKL, "rb"))
acc2ec = pickle.load(open(ACC2EC_PKL, "rb"))

multi_ec_labels = []
valid_indices = []

def expand_ec(ec):
    parts = ec.split(".")
    levels = []
    for i in range(1, 5):
        sub = ".".join(parts[:i])
        if "-" not in sub:
            levels.append(sub)
    return levels

for i, acc in enumerate(df["accessions"]):
    acc = acc[0] if isinstance(acc, list) else acc

    if acc in acc2ec:
        ecs = acc2ec[acc]
        if ecs:

            labels = []

            for ec in ecs:
                if "-" in ec:
                    continue
                labels.extend(expand_ec(ec))

            labels = list(set(labels))

            if len(labels) == 0:
                continue

            multi_ec_labels.append(labels)
            valid_indices.append(i)

print("Matched EC proteins:", len(valid_indices))

all_ec = set()
for labels in multi_ec_labels:
    all_ec.update(labels)

unique_ec = sorted(all_ec)
ec2idx = {ec: i for i, ec in enumerate(unique_ec)}

print("Number of EC classes:", len(unique_ec))

Y = torch.zeros(len(multi_ec_labels), len(unique_ec))

for i, labels in enumerate(multi_ec_labels):
    for ec in labels:
        Y[i, ec2idx[ec]] = 1

X = torch.stack([df["esm2"][i] for i in valid_indices])

print("X shape:", X.shape)
print("Y shape:", Y.shape)

torch.save({
    "X": X,
    "Y": Y,
    "ec2idx": ec2idx
}, "ec_dataset.pt")

print("Saved ec_dataset.pt")