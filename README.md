# Agriculture Assistant - LoRA Fine-Tuning Project

## Project Overview

This project fine-tunes **Qwen3-0.6B** using **LoRA (Low-Rank Adaptation)** to create an agricultural question-answering assistant.

The model was trained on a combination of:

- General agriculture QA data
- Synthetic agriculture QA data
- Domain-specific agriculture QA data

The objective is to provide accurate and practical answers related to agriculture while keeping the model lightweight and efficient for deployment.

---

## Base Model

- Model: `Qwen/Qwen3-0.6B`
- Fine-tuning method: LoRA
- Frameworks:
  - Transformers
  - PEFT
  - PyTorch
  - Hugging Face Datasets

---

## Dataset Sources

### 1. General Agriculture Dataset

**Source:** `talhakk/agriculture-qa`

Schema:

```json
{
  "question": "...",
  "answer": "..."
}
```

Role:

- General agricultural knowledge
- Basic farming concepts
- Crop and soil management

---

### 2. Synthetic Agriculture Dataset

**Source:** `KisanVaani/agriculture-qa-english-only`

Schema:

```json
{
  "question": "...",
  "answers": "..."
}
```

Role:

- Additional question diversity
- Synthetic QA generation
- Improved instruction-following behavior

---

### 3. Domain Agriculture Dataset

**Source:** `Satyam66/Agricare`

Schema:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

Role:

- Specialized agricultural knowledge
- Practical recommendations
- Real-world farming questions

---

## Data Processing Pipeline

### Step 1: Data Extraction

Raw datasets were downloaded directly from Hugging Face.

Each dataset was converted into a unified instruction format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

---

### Step 2: Cleaning

The following preprocessing steps were applied:

- Removed null values
- Removed empty questions
- Removed empty answers
- Trimmed extra whitespace
- Standardized text formatting

---

### Step 3: Chat Conversion

The instruction datasets were converted into chat format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert agricultural assistant."
    },
    {
      "role": "user",
      "content": "How do we define cover crop biomass?"
    },
    {
      "role": "assistant",
      "content": "A physical barrier between rainfall and the soil surface..."
    }
  ]
}
```

---

### Step 4: Dataset Splitting

Each dataset was split using:

| Split | Ratio |
|---------|---------|
| Train | 90% |
| Validation | 5% |
| Evaluation | 5% |

A combined evaluation set was also generated.

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Learning Rate | 1e-4 |
| Max Sequence Length | 1024 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |

---

## Training Strategy

Assistant-only training was used.

Loss was computed only on assistant responses while:

- System tokens were ignored
- User tokens were ignored
- Padding tokens were ignored

This allows the model to learn only the desired answer generation behavior.

---

## Results

The model successfully completed LoRA fine-tuning on the combined agricultural dataset.

Final training loss:

```text
1.412
```

The resulting adapter can be loaded on top of the original Qwen3-0.6B model for inference and deployment.

---

## Inference Example

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B"
)

model = PeftModel.from_pretrained(
    base_model,
    "Mumphone/agriculture-qwen3-lora"
)

tokenizer = AutoTokenizer.from_pretrained(
    "Mumphone/agriculture-qwen3-lora"
)
```

---

## Hugging Face Repository

Adapter Repository:

```text
Mumphone/agriculture-qwen3-lora
```

---

## Future Work

- Increase dataset size
- Add multilingual agricultural data
- Improve evaluation metrics
- Experiment with larger base models
- Integrate Retrieval-Augmented Generation (RAG)

---

---

## License

This project is intended for educational and research purposes.