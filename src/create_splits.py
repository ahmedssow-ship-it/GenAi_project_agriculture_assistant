import json
from pathlib import Path
from sklearn.model_selection import train_test_split

Path("data/splits").mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42


def load_json(path):

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def split_dataset(input_file, prefix):

    data = load_json(input_file)

    train_data, temp_data = train_test_split(
        data,
        test_size=0.10,
        random_state=SEED
    )

    validation_data, eval_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=SEED
    )

    save_json(
        f"data/splits/{prefix}_train.json",
        train_data
    )

    save_json(
        f"data/splits/{prefix}_validation.json",
        validation_data
    )

    save_json(
        f"data/splits/{prefix}_eval.json",
        eval_data
    )

    return eval_data


def chat_to_eval(row):

    instruction = ""
    reference = ""

    for msg in row["messages"]:

        if msg["role"] == "user":
            instruction = msg["content"]

        elif msg["role"] == "assistant":
            reference = msg["content"]

    return {
        "instruction": instruction,
        "input": "",
        "reference": reference
    }


all_eval = []

all_eval.extend(
    split_dataset(
        "data/chat_general.json",
        "chat_general"
    )
)

all_eval.extend(
    split_dataset(
        "data/chat_synthetic.json",
        "chat_synthetic"
    )
)

all_eval.extend(
    split_dataset(
        "data/chat_domain.json",
        "chat_domain"
    )
)


with open(
    "data/splits/eval_all.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for row in all_eval:

        eval_row = chat_to_eval(row)

        f.write(
            json.dumps(
                eval_row,
                ensure_ascii=False
            ) + "\n"
        )



print("Splits created.")