# Arithmetic LLM Adversarial Robustness
STAT 359 – Language Models & Word Embeddings Final Project

---

## Project Overview

Large language models can perform arithmetic reasoning, but their outputs may become unstable when exposed to adversarial feedback or misleading prompts.

This project investigates the **robustness of a small Arithmetic Language Model (Arithmetic LLM)** when subjected to adversarial pressure.

We study two main types of adversarial stress:

1. **Numeric perturbations**  
   Small numeric hints that attempt to push the model toward incorrect answers.

2. **Prompt pressure**  
   Changes in tone or authority designed to influence the model’s reasoning.

We also explore whether **increasing LoRA adapter rank** improves both arithmetic performance and resistance to adversarial perturbations.

The project combines **model training, adversarial evaluation, and robustness analysis** to better understand how architectural choices influence reasoning stability.

---

## Research Question

**How robust are arithmetic language models to adversarial numeric perturbations and prompt pressure, and can LoRA rank improve this robustness?**

We specifically examine:

- whether correct answers flip under adversarial hints  
- how flip rates scale with perturbation magnitude  
- whether LoRA rank improves clean arithmetic accuracy  
- whether higher LoRA rank reduces adversarial susceptibility  

---

## Repository Structure

```text
arithmetic_llm_stat359
│
├── README.md
├── LICENSE
├── pyproject.toml
├── poetry.lock
│
├── data/
│   ├── foundational_corpus.txt
│   ├── instruction_corpus.txt
│   └── tokenizer/
│
├── student/
│   └── final_project/
│       └── arithmetic_llm/
│           ├── transformer_model.py
│           ├── arithmetic_tokenizer.py
│           ├── arithmetic_verifier.py
│           ├── data_loader.py
│           ├── run_foundational_training.py
│           ├── run_instruction_training.py
│           ├── run_instruction_training_lora.py
│           ├── run_grpo_training.py
│           ├── run_evaluation.py
│           └── run_adversarial_numeric.py
│
├── experiments/
│   └── aggregate_lora_comparison.py
│
└── results/
    ├── comparison_results.csv
    ├── accuracy_vs_rank.png
    └── fliprate_vs_rank.png
```

---

## Environment Setup

This project uses **Poetry** for dependency management.

Install dependencies:

```bash
poetry install
```

Activate the environment:

```bash
poetry shell
```

Dependencies are defined in `pyproject.toml`.

---

## Reproducing the Full Training Pipeline

Run all commands from the **repository root**.

Replace the placeholder below with the timestamp directory created during training.

```text
YYYYMMDD_HHMMSS
```

Example:

```text
models/foundational_20260201_012912_173614/best_model.pt
```

---

## Step 1 – Corpus Generation

Generate the foundational arithmetic corpus:

```bash
poetry run python -m instructor.final_project.arithmetic_llm.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt
```

Generate instruction training corpus:

```bash
poetry run python -m instructor.final_project.arithmetic_llm.generate_instruction_corpus_mixed \
  --num-samples 20000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.0 \
  --output-mixed data/instruction_corpus.txt
```

---

## Step 2 – Train the Tokenizer

```bash
poetry run python -m instructor.final_project.arithmetic_llm.train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000
```

---

## Step 3 – Sequence Length Analysis

```bash
poetry run python -m instructor.final_project.arithmetic_llm.check_sequence_lengths \
  --corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --corpus-type instruction
```

---

## Step 4 – Foundational Model Training

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_foundational_training \
  --corpus-path data/foundational_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --max-seq-length 512 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --num-epochs 5
```

---

## Step 5 – Foundational Model Evaluation

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1
```

---

## Step 6 – Instruction Fine-Tuning

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 5 \
  --batch-size 16 \
  --learning-rate 0.0001
```

---

## Step 7 – Instruction Model Evaluation

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

---

## Step 8 – LoRA Fine-Tuning

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_instruction_training_lora \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 3 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules attention \
  --save-merged-model
```

---

## Step 9 – LoRA Model Evaluation

```bash
poetry run python -m instructor.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

---

## Step 10 – Adversarial Numeric Evaluation

```bash
poetry run python -m student.final_project.arithmetic_llm.run_adversarial_numeric \
  --baseline-csv results/baseline_table.csv \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --output-dir results/lora_numeric_eval
```

---

## Aggregating Final Results

```bash
python experiments/aggregate_lora_comparison.py
```

This generates:

```text
results/comparison_results.csv
results/accuracy_vs_rank.png
results/fliprate_vs_rank.png
```

---

## Key Results

| LoRA Rank | Clean Accuracy |
|-----------|---------------|
| 4 | 40.0% |
| 8 | 47.5% |
| 16 | 49.0% |

Higher LoRA ranks improved arithmetic performance and reduced adversarial flip rates.

---

## Output Management

Experiment outputs are stored in:

```text
results/
```

Training checkpoints and logs are generated in:

```text
models/
```

Model checkpoints are **not included in the repository**.

---

## License

MIT License

---

## Contributors

### Kai Neumark

- adversarial numeric experiment design  
- LoRA robustness analysis  
- experiment pipeline and evaluation  
- results aggregation and visualization  

### Andrew Ross

- Arithmetic LLM training pipeline  
- foundational model architecture  
- tokenizer and dataset generation  
- model training infrastructure  
