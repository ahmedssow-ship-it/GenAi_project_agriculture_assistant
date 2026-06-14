import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert agricultural assistant. "
    "Provide accurate, practical, and concise answers about farming, crops, "
    "soil management, irrigation, pests, diseases, fertilizers, livestock, "
    "and sustainable agriculture."
).strip()


def load_jsonl(path):

    rows = []

    with open(path, encoding="utf-8") as f:

        for line in f:
            rows.append(json.loads(line))

    return rows


def save_json(path, data):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def convert(source_file, output_file):

    rows = load_jsonl(source_file)

    chat_rows = []

    for row in rows:

        user_content = row["instruction"]

        if row["input"]:
            user_content += f"\n\nContext:\n{row['input']}"

        chat_rows.append({
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": row["output"]
                }
            ]
        })

    save_json(output_file, chat_rows)


convert(
    "data/source_1_general.jsonl",
    "data/chat_general.json"
)

convert(
    "data/source_2_synthetic.jsonl",
    "data/chat_synthetic.json"
)

convert(
    "data/source_3_domain.jsonl",
    "data/chat_domain.json"
)

print("Chat datasets created.")