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
   Changes in tone or authority designed to influence the model's reasoning.

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
│           ├── __init__.py
│           ├── arithmetic_tokenizer.py
│           ├── arithmetic_verifier.py
│           ├── check_sequence_lengths.py
│           ├── corpus_generator.py
│           ├── data_loader.py
│           ├── diagnose_speed.py
│           ├── evaluator.py
│           ├── generate_corpus.py
│           ├── generate_foundational_plaintext.py
│           ├── generate_instruction_corpus_mixed.py
│           ├── generator.py
│           ├── grpo_config.py
│           ├── grpo_trainer.py
│           ├── interactive_solver.py
│           ├── lora_config.py
│           ├── lora_layer.py
│           ├── lora_utils.py
│           ├── merge_lora_adapter.py
│           ├── print_token_table.py
│           ├── profile_training.py
│           ├── run_evaluation.py
│           ├── run_evaluator_tests.py
│           ├── run_foundational_training.py
│           ├── run_grpo_training.py
│           ├── run_instruction_training.py
│           ├── run_instruction_training_lora.py
│           ├── run_interactive.py
│           ├── show_operator_hardcoding.py
│           ├── show_token_table.py
│           ├── test_eos_truncation.py
│           ├── train_foundational.py
│           ├── train_grpo.py
│           ├── train_instruction.py
│           ├── train_instruction_lora.py
│           ├── train_tokenizer.py
│           ├── training_config.py
│           └── transformer_model.py
│
├── experiments/
│   ├── README.md
│   ├── analyze_adversarial_results.py
│   ├── generate_additional_presentation_plots.py
│   ├── generate_baseline_table.py
│   ├── generate_cross_axis_wrong_revision_heatmaps.py
│   ├── generate_presentation_artifacts.py
│   ├── split_baseline_adv_def_grouped_plot.py
│   ├── split_cross_axis_grouped_plots.py
│   ├── split_cross_axis_grouped_plots_strict.py
│   └── split_cross_axis_wrong_revision_faceted_strict.py
│
└── results/
    ├── accuracy_vs_rank.png
    ├── cross_axis_accuracy_language.png
    ├── cross_axis_accuracy_numeric.png
    ├── cross_axis_accuracy_politeness.png
    ├── flip_rate_by_difficulty_and_offset_20260306_190407.png
    ├── fliprate_vs_rank.png
    ├── plot1_baseline_accuracy_by_difficulty.png
    ├── plot2_flip_rate_vs_numeric_perturbation.png
    ├── plot4_parse_success_by_difficulty.png
    └── training_val_loss_overlay.png
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

```bash
poetry run python -m student.final_project.arithmetic_llm.generate_foundational_plaintext \
  --num-samples 100000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.05 \
  --output-txt data/foundational_corpus.txt
```

```bash
poetry run python -m student.final_project.arithmetic_llm.generate_instruction_corpus_mixed \
  --num-samples 20000 \
  --max-depth 4 \
  --num-range 1 20 \
  --invalid-rate 0.0 \
  --output-mixed data/instruction_corpus.txt
```

---

## Step 2 – Train the Tokenizer

```bash
poetry run python -m student.final_project.arithmetic_llm.train_tokenizer \
  --corpus-path data/foundational_corpus.txt \
  --output-dir data/tokenizer \
  --vocab-size 1000
```

---

## Step 3 – Sequence Length Analysis

```bash
poetry run python -m student.final_project.arithmetic_llm.check_sequence_lengths \
  --corpus-path data/instruction_corpus.txt \
  --tokenizer-path data/tokenizer \
  --corpus-type instruction
```

---

## Step 4 – Foundational Model Training

```bash
poetry run python -m student.final_project.arithmetic_llm.run_foundational_training \
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
poetry run python -m student.final_project.arithmetic_llm.run_evaluation \
  --model-path models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 100 \
  --batch-size 1
```

---

## Step 6 – Instruction Fine-Tuning

```bash
poetry run python -m student.final_project.arithmetic_llm.run_instruction_training \
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
poetry run python -m student.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

---

## Step 8 – LoRA Fine-Tuning

```bash
poetry run python -m student.final_project.arithmetic_llm.run_instruction_training_lora \
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
poetry run python -m student.final_project.arithmetic_llm.run_evaluation \
  --model-path models/instruction_lora_YYYYMMDD_HHMMSS/merged_model.pt \
  --tokenizer-path data/tokenizer \
  --max-gen-length 512 \
  --num-samples 1000 \
  --batch-size 1
```

---

## Step 10 – GRPO Training

```bash
poetry run python -m student.final_project.arithmetic_llm.run_grpo_training \
  --instruction-corpus-path data/instruction_corpus.txt \
  --output-dir models/ \
  --tokenizer-path data/tokenizer \
  --foundational-checkpoint models/foundational_YYYYMMDD_HHMMSS/best_model.pt \
  --num-epochs 3
```

---

## Step 11 – Adversarial Numeric Evaluation

```bash
python experiments/generate_baseline_table.py
```

```bash
python experiments/analyze_adversarial_results.py
```

```bash
python experiments/generate_presentation_artifacts.py
```

---

## Aggregating Final Results

```bash
python experiments/generate_additional_presentation_plots.py
python experiments/split_baseline_adv_def_grouped_plot.py
python experiments/split_cross_axis_grouped_plots.py
python experiments/split_cross_axis_grouped_plots_strict.py
python experiments/split_cross_axis_wrong_revision_faceted_strict.py
python experiments/generate_cross_axis_wrong_revision_heatmaps.py
```

---

## Key Results

### LoRA Performance

| LoRA Rank | Clean Accuracy |
|-----------|---------------|
| 4         | 40.0%         |
| 8         | 47.5%         |
| 16        | 49.0%         |

Higher LoRA ranks improve arithmetic performance.

---

### Baseline Accuracy by Difficulty

| Difficulty | Correct | Total | Accuracy (%) |
|------------|---------|-------|--------------|
| Easy       | 199     | 201   | 99.0%        |
| Medium     | 53      | 80    | 66.3%        |
| Hard       | 58      | 219   | 26.5%        |

---

### Adversarial Flip Rate by Perturbation

| Perturbation | Flips | Total | Flip Rate (%) |
|--------------|-------|-------|----------------|
| off_by_1     | 118   | 310   | 38.1%          |
| off_by_2     | 117   | 310   | 37.7%          |
| off_by_5     | 151   | 310   | 48.7%          |
| off_by_10    | 159   | 310   | 51.3%          |
| random       | 162   | 310   | 52.3%          |

---

### Key Insights

- Accuracy drops sharply with difficulty (99% → 26%)
- Even small perturbations flip ~38% of correct answers
- Larger perturbations exceed 50% failure rates
- LoRA improves accuracy but does not eliminate adversarial vulnerability

---

### Interpretation

The model demonstrates strong arithmetic capability but weak robustness:

- Correct reasoning can be overridden by small misleading hints  
- Errors arise from instability in reasoning, not lack of knowledge  
- Robustness does not scale proportionally with model improvements  

---

### Example Failure

- Input: `7 + 5`  
- Correct: `12`  
- Perturbation: "closer to 13"  
- Output: **13 (incorrect)**  

---

## Output Management

Experiment outputs are stored in:

```text
results/
```

Training checkpoints are stored in:

```text
models/
```

Model files are excluded from the repository.

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

---

## Credits

This project was built on top of the STAT 359 course repository and draws on prior work in parameter-efficient fine-tuning and mathematical reasoning robustness.

### Base Repository

- Lizhen0909, *STAT 359 Course Repository*:  
  <https://github.com/Lizhen0909/stat359>

### References

1. Hu, E. J., Shen, Y., Wallis, P., et al. (2022).  
   *LoRA: Low-rank adaptation of large language models.*  
   Proceedings of ICLR 2022.  
   <https://arxiv.org/abs/2106.09685>

2. Li, Q., Cui, L., Zhao, X., Kong, L., & Bi, W. (2024).  
   *GSM-Plus: A comprehensive benchmark for evaluating the robustness of LLMs as mathematical problem solvers.*  
   Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), pp. 2961–2984.  
   <https://arxiv.org/abs/2402.19255>

3. Mirzadeh, I., Alizadeh, K., Shahrokhi, H., et al. (2025).  
   *GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models.*  
   Proceedings of ICLR 2025.  
   <https://arxiv.org/abs/2410.05229>
