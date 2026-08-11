# Adaptive Player-Behavior Classification and NPC Simulation

A research prototype for predicting player behavioral archetypes from gameplay
telemetry using supervised machine learning, and for mapping those predictions
onto adaptive non-player-character (NPC) strategies evaluated in a lightweight,
turn-based combat simulation.

> **Scope note.** This is a research/academic project. The dataset is **synthetic**
> and does **not** represent real players. The NPC simulation is a research
> prototype, not a game engine, and it is **not** evidence of classifier accuracy.
> ML evaluation and NPC simulation are treated as separate, deliberately
> independent stages in this report.

---

## 1. Project Title

**Adaptive Player-Behavior Classification and NPC Simulation**

- ML stage: telemetry → player-behavior class (Aggressive, Balanced, Defensive, Explorer)
- Adaptation stage: predicted behavior → NPC strategy parameters
- Simulation stage: turn-based combat driven by those strategy parameters

---

## 2. Research Motivation / Problem

In many games, NPC opponents behave identically toward all players. This is
often perceived as artificial and unengaging. A more compelling design reacts to
*how a player actually plays*: aggressive players could face more defensive,
counter-attacking opponents; explorers could be ambushed near objectives.

The core problem is that player behavior is not directly observable from a few
low-level signals; it must be inferred. This project asks:

> Can a machine-learning classifier trained on gameplay telemetry predict a
> player's behavioral archetype well enough to drive an adaptive NPC response,
> in a way that is reproducible and explainable?

This is formulated as a **multiclass classification** problem followed by a
**rule-based adaptation + simulation** layer. It deliberately uses classical,
interpretable ML rather than reinforcement learning, so the entire pipeline can
be audited from telemetry to NPC decision.

---

## 3. Research Objective

1. Generate a controlled, non-trivially-separable **synthetic telemetry dataset**
   with four behavioral classes and known class overlap.
2. Build and evaluate a set of classical classifiers
   (Logistic Regression, RBF SVM, Random Forest) under a strict
   train/validation/test protocol with **no leakage**.
3. Quantify generalization via **5-fold stratified cross-validation**, then
   **hyperparameter tuning**, then a single **held-out test evaluation**.
4. Map predicted behavior to **NPC strategy parameters** through a deterministic
   adaptation rule (no retraining at runtime).
5. Demonstrate the full loop
   *telemetry → prediction → strategy → simulation* on individual training
   samples and report the resulting simulated encounters.

---

## 4. System Overview

```
         GAMEPLAY TELEMETRY                     (14 features)
                    |
                    v
        ML BEHAVIOR PREDICTION                  (Logistic Regression pipeline)
                    |
                    v
        PREDICTED BEHAVIOR                      (Aggressive / Balanced / Defensive / Explorer)
                    |
                    v
        NPC STRATEGY                            (posture / distance / combat_style / aggression_level)
                    |
                    v
        TURN-BASED SIMULATION                   (deterministic, seeded, 10-turn cap)
```

The ground-truth label of a demonstration sample is used **only for reporting**.
The NPC strategy is always derived from the **model's predicted** behavior.

---

## 5. Dataset Description

- **Total samples:** 10,000 (2,500 per class)
- **Generation:** synthetic multivariate-Gaussian samples per class with a
  hand-designed class mean profile, a covariance matrix encoding realistic
  feature correlations, and Gaussian noise.
- **Controls** (from `config.yaml`): `n_samples_per_class=2500`,
  `noise_level=0.3`, `class_overlap=0.25`.
- Class mean profiles are designed so that classes are **not trivially
  separable** (e.g., Aggressive is high in combat/damage/risk and low in
  defense/exploration; Defensive is the opposite; Balanced sits in the middle).

> **Important.** All data is procedurally generated. It is used to validate the
> *methodology* (pipeline, evaluation protocol, adaptation logic), not to claim
> results on real player populations.

## 6. Fourteen Telemetry Features

All features are normalized to `[0, 1]`. Order and naming come from the single
source of truth in `src/features.py`.

| # | Feature | Interpretation |
|---|---------|----------------|
| 1 | `combat_frequency` | How often the player engages in combat |
| 2 | `damage_dealt` | Average damage per encounter |
| 3 | `damage_taken` | Average damage received |
| 4 | `kill_count` | Enemies defeated per session |
| 5 | `death_count` | Times the player died |
| 6 | `exploration_rate` | Map coverage percentage |
| 7 | `distance_traveled` | Total units moved |
| 8 | `resource_collection` | Resources gathered per minute |
| 9 | `ability_usage` | Special-ability activations per minute |
| 10 | `risk_taking` | High-risk actions (low-health fights, etc.) |
| 11 | `defensive_actions` | Blocking, dodging, shielding frequency |
| 12 | `objective_focus` | Primary-objective completion rate |
| 13 | `social_interactions` | Co-op/trade/chat frequency |
| 14 | `session_duration` | Minutes per play session |

The generator encodes correlations such as combat↔damage↔kills,
exploration↔distance↔resources, risk↔(−)defense, and combat↔(−)exploration.

## 7. Four Behavioral Classes

| Class | Signature in telemetry |
|-------|------------------------|
| **Aggressive** | High combat, damage, kills, risk; low defense, low exploration |
| **Balanced** | Moderate values across all dimensions |
| **Defensive** | Low combat/risk; high defensive actions; moderate exploration |
| **Explorer** | Low combat; high exploration, distance, resources; moderate social |

Classes are intentionally overlapping (particularly **Defensive ↔ Balanced**,
which is confirmed by the results below).

## 8. Data Preprocessing and Train/Validation/Test Split

Protocol implemented in `src/features.py` (`prepare_features_and_split`):

1. **Split first, then scale/encode** — prevents data leakage.
2. **Split 1:** 80/20 train+val vs **test** (`test_size=0.20`, stratified).
3. **Split 2:** train vs **validation** from the remaining 80%
   (`val_ratio = 0.15 / 0.8 = 0.1875`, stratified).
4. Labels encoded with `LabelEncoder` fit **on the training split only**.
5. Features standardized with `StandardScaler` fit **on the training split only**.

| Split | Samples |
|-------|---------|
| Train | **6,500** |
| Validation | **1,500** |
| Test (held out) | **2,000** (500 per class) |

**Discipline:** the held-out **2,000-sample test set was never used for
training, cross-validation, hyperparameter tuning, or the demonstrations.** It is
used exactly once, in the final evaluation (Section 12). Cross-validation and
tuning operate on the 6,500 training samples only. The validation set (1,500) is
used for the initial baseline comparison.

## 9. Models Evaluated

Three models, all trained on the scaled training data (see `src/models.py`,
`src/run_baseline_training.py`, `src/run_cv_analysis.py`):

| Model | Key configuration (baseline) |
|-------|------------------------------|
| **Logistic Regression** | `C=1.0`, `l2`, `lbfgs`, `max_iter=1000` |
| **SVM (RBF)** | `C=1.0`, `gamma='scale'` |
| **Random Forest** | 200 trees, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2` |

> Note: `src/models.py` also defines an **XGBoost** configuration, but no
> XGBoost model, tuning, or evaluation artifacts exist in this repository; it was
> not part of the reported experiments.

Baseline validation results (`models/model_comparison.csv`, evaluated on the
1,500-sample validation split):

| Model | Val Accuracy | Val F1 |
|-------|-------------|--------|
| Logistic Regression | 0.7533 | 0.7488 |
| SVM | 0.7533 | 0.7497 |
| Random Forest | 0.7400 | 0.7364 |

## 10. Five-Fold Cross-Validation Methodology

Performed by `src/run_cv_analysis.py` on the **6,500 training samples only**:

- **StratifiedKFold**, `n_splits=5`, `shuffle=True`, `random_state=42`.
  Each fold is 1,300 samples; the model trains on 5,200 and validates on the fold.
- Models are wrapped in pipelines containing the `StandardScaler`, so scaling is
  refit **inside each fold** (no fold leakage).
- Metrics: accuracy, weighted precision, weighted recall, weighted F1
  (5 fold scores + mean ± std each).

Baseline (untuned) CV results (`models/cv_summary.csv`,
`models/cross_validation_results.csv`):

| Model | Accuracy (mean ± std) | Weighted F1 (mean ± std) |
|-------|-----------------------|---------------------------|
| Logistic Regression | 0.7532 ± 0.0067 | 0.7512 ± 0.0080 |
| SVM | 0.7429 ± 0.0041 | 0.7416 ± 0.0046 |
| Random Forest | 0.7363 ± 0.0052 | 0.7331 ± 0.0055 |

CV accuracy is stable across folds (std ≈ 0.4–0.8%), indicating consistent
behavior across data subsets.

## 11. Hyperparameter Tuning Methodology

Performed by `src/run_hyperparameter_tuning.py` with **GridSearchCV** on the
**6,500 training samples only**, again 5-fold stratified CV, scoring =
**weighted F1**.

| Model | Search space | Combinations | Best parameters |
|-------|-------------|--------------|------------------|
| Logistic Regression | `C ∈ {0.01, 0.1, 1, 10, 100}`, `penalty=l2` | 5 | `C=0.1` |
| SVM | `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, 0.01, 0.1, 1}`, `kernel=rbf` | 16 | `C=1.0`, `gamma=0.01` |
| Random Forest | `n_estimators ∈ {100, 300}`, `max_depth ∈ {None, 10, 20}`, `min_samples_split ∈ {2, 5}`, `min_samples_leaf ∈ {1, 2}` | 24 | `max_depth=20`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=300` |

Baseline vs tuned CV (`models/tuning_comparison.csv`):

| Model | Baseline CV Acc | Tuned CV Acc | Δ Acc | Baseline CV F1 | Tuned CV F1 | Δ F1 |
|-------|-----------------|--------------|-------|----------------|-------------|------|
| Logistic Regression | 0.7532 | 0.7537 | +0.0005 | 0.7512 | 0.7515 | +0.0003 |
| SVM | 0.7429 | 0.7552 | +0.0123 | 0.7416 | 0.7532 | +0.0116 |
| Random Forest | 0.7363 | 0.7354 | −0.0009 | 0.7331 | 0.7313 | −0.0018 |

Tuning gave the largest improvement to the SVM. Random Forest did not improve
within the chosen grid (slight decrease). Best hyperparameters are saved in
`models/best_hyperparameters.csv`.

## 12. Final Held-Out Test Results

Single evaluation on the **2,000 held-out test samples**, using the **tuned**
models (`src/run_final_test_evaluation.py`, results in
`models/final_model_comparison.csv`):

| Model | CV Acc | Test Acc | CV F1 | Test F1 | Test Precision | Test Recall |
|-------|--------|----------|-------|---------|----------------|-------------|
| Logistic Regression | 0.7537 | **0.7600** | 0.7515 | **0.7565** | 0.7549 | 0.7600 |
| SVM | 0.7552 | **0.7600** | 0.7532 | 0.7563 | 0.7546 | 0.7600 |
| Random Forest | 0.7354 | 0.7455 | 0.7313 | 0.7413 | 0.7399 | 0.7455 |

- Logistic Regression and SVM tie at **76.0% test accuracy**; LR has the
  marginally higher weighted F1 (0.7565).
- CV and test figures are close (Δ accuracy ≤ 0.010), showing stable
  generalization with little overfitting.
- **The deployed inference model** (`models/logistic_regression_tuned.joblib`,
  used by `src/inference.py` and both demos) is the **tuned Logistic Regression
  with `C=0.1`**.

## 13. Per-Class Performance

Per-class precision/recall/F1 on the 2,000-sample test set
(`models/final_test_per_class.csv`):

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| Logistic Regression | Aggressive | 0.8147 | 0.8620 | 0.8377 |
| Logistic Regression | Balanced | 0.6239 | 0.5540 | 0.5869 |
| Logistic Regression | Defensive | 0.7963 | 0.8520 | 0.8232 |
| Logistic Regression | Explorer | 0.7846 | 0.7720 | 0.7782 |
| SVM | Aggressive | 0.8127 | 0.8680 | 0.8395 |
| SVM | Balanced | 0.6259 | 0.5520 | 0.5866 |
| SVM | Defensive | 0.7959 | 0.8500 | 0.8221 |
| SVM | Explorer | 0.7841 | 0.7700 | 0.7770 |
| Random Forest | Aggressive | 0.7835 | 0.8540 | 0.8172 |
| Random Forest | Balanced | 0.6106 | 0.5300 | 0.5675 |
| Random Forest | Defensive | 0.7780 | 0.8340 | 0.8050 |
| Random Forest | Explorer | 0.7876 | 0.7640 | 0.7756 |

**Key pattern.** Aggressive and Defensive are the most separable classes
(F1 ≈ 0.82–0.84), Explorer is moderate (≈ 0.78), and **Balanced is the hardest
class (F1 ≈ 0.57–0.59)**. This is a direct consequence of the synthetic class
overlap between Balanced and the other archetypes — in particular,
Defensive players are frequently predicted as Balanced (see the demonstration in
Section 16). Confusion matrices are saved under `reports/figures/`.

## 14. Adaptive NPC Architecture

Implemented in `src/adaptive_npc.py`. Given a **predicted** behavior, a
deterministic rule returns a strategy dictionary; no ML runs at adaptation time.

| Predicted behavior | postures | distance | combat_style | aggression_level | patrol_radius | objective_awareness |
|--------------------|----------|----------|--------------|------------------|---------------|---------------------|
| Aggressive | defensive | maintain_range | counterattack | 0.3 | normal | — |
| Balanced | mixed | standard | standard | 0.5 | normal | — |
| Defensive | offensive | close_gap | flanking | 0.8 | normal | — |
| Explorer | adaptive | dynamic | ambush_at_objectives | 0.5 | expanded | high |

Design rationale:

- **Aggressive** players (high pressure) face a *defensive* NPC that keeps its
  distance and punishes attacks with **counterattacks**.
- **Defensive** players (low pressure) are challenged by an *offensive* NPC that
  **closes the gap** and **flanks**.
- **Explorer** players are met by an *adaptive* NPC with expanded patrols and
  **ambushes at objectives**.
- **Balanced** players face a standard mixed strategy.

The `npc_strategy` dictionary is the single contract between adaptation and
simulation (`triggered_by` records which predicted behavior produced it).

## 15. End-to-End ML → NPC Simulation Pipeline

Implemented in `src/simulation.py` and `src/simulation_demo.py`.

**Simulation engine** (`simulation.py`) — a deterministic, turn-based research
prototype (stdlib only, no dependency on the trained model):

- `SimulationState` dataclass: `turn`, `player_hp`, `npc_hp`, `distance`
  (`close/medium/far`), `player_action`, `npc_action`, `player_behavior`,
  `npc_strategy`, `log`.
- Scripted player actions per behavior: Aggressive → `attack/charge/risky_attack`;
  Balanced → `attack/defend/reposition`; Defensive → `defend/retreat/block`;
  Explorer → `move_to_objective/scout/ambush`.
- NPC decisions consume the `npc_strategy` dict directly (posture, distance
  preference, combat style, aggression).
- Simple combat resolution: `attack vs block → 0`, `attack vs dodge → 50%`,
  `attack vs retreat → hit`, flank `1.5×`, ambush at objectives `1.5×`,
  counterattack after a blocked player attack.
- **Seeded RNG** (`random.Random(seed)`) ⇒ reproducible turn-by-turn logs.
- Runs until one side reaches 0 HP or a configurable max-turns cap.

**Demo runner** (`simulation_demo.py`) — end-to-end flow for four demonstration
players (one per ground-truth class), sampled **from the 6,500 training rows only**:

```
ground truth (reporting only)
telemetry → ML prediction → predicted behavior → get_npc_strategy() → simulation
```

For each player it prints ground truth, prediction, confidence, the NPC
strategy, the full turn-by-turn log, final HP, turn count, and result, then a
summary table. It asserts that `strategy['triggered_by'] == predicted behavior`,
i.e., the strategy is provably derived from the **prediction**, never the label.

## 16. Simulation Demonstration Results

Output of `python -m src.simulation_demo` (fixed seed `42`, max 10 turns per
encounter). Players are individual **training** samples, selected deterministically
(preferring, per class, a sample the model predicts correctly where one exists).

| Player | Ground truth | Prediction | Confidence | NPC Strategy | Result (seed 42, ≤10 turns) |
|--------|--------------|------------|------------|--------------|------------------------------|
| P1 | Aggressive | Aggressive | 0.6797 | defensive / maintain_range / counterattack | Time limit — player 28 HP, NPC 10 HP |
| P2 | Balanced | Balanced | 0.7223 | mixed / standard / standard | Time limit — player 55 HP, NPC 80 HP |
| P3 | Defensive | **Balanced** | 0.7000 | mixed / standard / standard | Time limit — player 55 HP, NPC 80 HP |
| P4 | Explorer | Explorer | 0.5166 | adaptive / dynamic / ambush_at_objectives | NPC wins in 5 turns — player 0 HP, NPC 100 HP |

Observations:

- P1 shows the **counterattack** style: the defensive NPC repeatedly blocked
  and riposted, keeping the fight close.
- P4 shows the **ambush-at-objectives** style: once the simulated player moved
  toward objectives, the NPC's 1.5× ambush attacks (27 damage) ended the
  encounter in 5 turns.
- **P3 is the intended "class overlap" demonstration.** The ground-truth
  Defensive player is predicted as Balanced (recall the Balanced class has the
  lowest F1, ≈0.59), so the NPC adopts the Balanced mixed strategy rather than
  the offensive/flanking one. This illustrates that the pipeline adapts to the
  **prediction**, and that imperfect classification directly changes the
  resulting NPC behavior.

> **Interpretation guardrail.** These simulated encounters demonstrate that the
> pipeline executes and that distinct predictions yield distinct NPC strategies.
> They are **not** a measure of classifier quality and do **not** imply anything
> about real gameplay balance. The ML evidence is Sections 10–13; the simulation
> is a separate prototype.

## 17. Limitations

1. **Synthetic data.** Results are on procedurally generated telemetry. They
   establish the methodology but **cannot be generalized to real players** without
   real telemetry.
2. **Class overlap.** The generator intentionally overlaps classes. Balanced is
   the hardest class (test F1 ≈ 0.59); Defensive players are commonly predicted
   as Balanced, as seen for P3 in Section 16. This is a property of the data, not
   a bug, but it caps expected accuracy.
3. **Simulation ≠ validation.** The turn-based simulation is a lightweight
   research prototype (scripted player, hand-tuned damage numbers, seeded
   randomness). It does **not** prove the classifier is good and does not claim
   gameplay realism.
4. **Rule-based adaptation only.** NPC strategy is a fixed deterministic lookup;
   there is no learning or tuning of the adaptation layer, and no
   reinforcement learning.
5. **Single seed / fixed protocol.** All randomness is seeded (`random_seed: 42`),
   which makes results reproducible but means variance across seeds is not
   characterized.
6. **XGBoost not evaluated.** Although configured in `models.py`, XGBoost has no
   saved results here and is excluded from the reported comparisons.
7. **Sensitivity of demo samples.** Demonstration players are selected from
   training samples, so their predictions are illustrative rather than
   representative of test-time behavior.

## 18. Reproducibility / Setup Instructions

The project requires Python 3.x and the dependencies listed in
`requirements.txt`.

```bash
# 1. Clone the repository and enter the project directory

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the synthetic dataset
python -m src.data_generation

# 5. Reproduce the ML pipeline in order
python -m src.run_baseline_training
python -m src.run_svm_training
python -m src.run_cv_analysis
python -m src.run_hyperparameter_tuning
python -m src.run_final_test_evaluation

# 6. Run the demonstrations
python -m src.demo
python -m src.simulation_demo
```

**The project uses a fixed random seed (`42`) for reproducibility. Re-running the pipeline with the same software environment and configuration should reproduce the reported results, subject to library/version differences.**

## 19. Project Structure

```
Player_Behaviour/
├── config.yaml                     # Seeds, data-gen, features, models, CV, paths
├── requirements.txt
├── data/
│   └── raw/synthetic_telemetry.csv # 10,000 generated samples (14 features + label)
├── src/
│   ├── data_generation.py          # Synthetic telemetry generator
│   ├── features.py                 # FEATURE_NAMES, CLASS_NAMES, split + scaling
│   ├── models.py                   # Model definitions (LR, SVM, RF, XGBoost config)
│   ├── run_baseline_training.py    # Baseline LR/RF + validation comparison
│   ├── run_svm_training.py         # Baseline SVM
│   ├── run_cv_analysis.py          # 5-fold stratified CV
│   ├── run_hyperparameter_tuning.py# GridSearchCV tuning, saves tuned models
│   ├── run_final_test_evaluation.py# Single held-out test evaluation
│   ├── evaluation.py               # Metrics, CV, plotting utilities
│   ├── inference.py                # BehaviorPredictor (loads tuned LR pipeline)
│   ├── adaptive_npc.py             # Behavior -> NPC strategy rule
│   ├── simulation.py               # Turn-based NPC simulation engine
│   ├── demo.py                     # Inference -> adaptation demo
│   ├── simulation_demo.py          # End-to-end telemetry -> simulation demo
│   └── utils/paths.py              # Project-root-relative path helpers
├── models/                         # Saved joblibs + all result CSVs (see below)
├── reports/figures/                # Generated figures (EDA, CV, tuning, confusion)
└── notebooks/01_exploratory_analysis.ipynb
```

Key saved artifacts in `models/`:

| File | Contents |
|------|----------|
| `model_comparison.csv` | Baseline validation accuracy / F1 |
| `cross_validation_results.csv`, `cv_summary.csv` | Per-fold and mean CV metrics |
| `lr_tuning_results.csv`, `svm_tuning_results.csv`, `rf_tuning_results.csv` | Full GridSearchCV results |
| `best_hyperparameters.csv`, `tuning_comparison.csv` | Best params and baseline-vs-tuned deltas |
| `final_model_comparison.csv`, `final_test_per_class.csv` | Held-out test and per-class results |
| `logistic_regression_tuned.joblib` | **Deployed inference pipeline** (scaler + LR, C=0.1) |
| `svm_tuned.joblib`, `random_forest_tuned.joblib` | Tuned comparison models |

## 20. Future Work

1. **Real telemetry.** Collect anonymized gameplay data from an actual game to
   re-validate the four-class taxonomy and the adaptation rules.
2. **Richer classes.** Add classes such as *social/cooperative* or *speedrunner*,
   and re-examine the Balanced/Defensive boundary (e.g., via class weights or
   better features).
3. **Better models.** Evaluate XGBoost and modern approaches, and test whether
   calibration of `predict_proba` improves confidence-based adaptation.
4. **Adaptation tuning.** Replace the fixed rule lookup with an
   optimization/AB-test loop over strategy parameters, while keeping decisions
   interpretable.
5. **Simulation fidelity.** Add spatial maps, objectives, and stochastic
   opponent behavior; characterize sensitivity across multiple seeds rather than
   a single seed.
6. **Online adaptation.** Stream telemetry and re-predict behavior periodically so
   the NPC strategy can follow a changing player style mid-session.
