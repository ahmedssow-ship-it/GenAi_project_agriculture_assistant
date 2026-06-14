from datasets import load_dataset
import json
from pathlib import Path

Path("data").mkdir(exist_ok=True)

SEED = 42

N_GENERAL = 10000
N_SYNTHETIC = 5000
N_DOMAIN = 5000

#N_GENERAL = 100
#N_SYNTHETIC = 100
#N_DOMAIN = 100


def save_jsonl(dataset, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ==========================
# SOURCE 1 : GENERAL
# ==========================

agricare = load_dataset("Satyam66/Agricare")["train"]

agricare = (
    agricare
    .shuffle(seed=SEED)
    .select(range(min(N_GENERAL, len(agricare))))
)
N_SYNTHETIC
general_rows = []

for row in agricare:

    general_rows.append({
        "instruction": str(row["instruction"]).strip(),
        "input": str(row["input"]).strip(),
        "output": str(row["output"]).strip(),
        "source": "general",
        "source_detail": "Satyam66/Agricare"
    })

save_jsonl(
    general_rows,
    "data/source_1_general.jsonl"
)

# ==========================
# SOURCE 2 : SYNTHETIC
# ==========================

kisan = load_dataset(
    "KisanVaani/agriculture-qa-english-only"
)["train"]

kisan = (
    kisan
    .shuffle(seed=SEED)
    .select(range(min(N_SYNTHETIC, len(kisan))))
)

synthetic_rows = []

for row in kisan:

    synthetic_rows.append({
        "instruction": str(row["question"]).strip(),
        "input": "",
        "output": str(row["answers"]).strip(),
        "source": "synthetic",
        "source_detail": "KisanVaani/agriculture-qa-english-only"
    })

save_jsonl(
    synthetic_rows,
    "data/source_2_synthetic.jsonl"
)

# ==========================
# SOURCE 3 : DOMAIN
# ==========================

talhakk = load_dataset(
    "talhakk/agriculture-qa"
)["train"]

talhakk = (
    talhakk
    .shuffle(seed=SEED)
    .select(range(min(N_DOMAIN, len(talhakk))))
)

domain_rows = []

for row in talhakk:

    domain_rows.append({
        "instruction": str(row["question"]).strip(),
        "input": "",
        "output": str(row["answer"]).strip(),
        "source": "domain",
        "source_detail": "talhakk/agriculture-qa"
    })

save_jsonl(
    domain_rows,
    "data/source_3_domain.jsonl"
)

print("Done.")