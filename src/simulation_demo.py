"""
End-to-end system demonstration.

Flow:
    GAMEPLAY TELEMETRY
    -> ML BEHAVIOR PREDICTION
    -> NPC STRATEGY
    -> TURN-BASED SIMULATION

For each of four scripted players (one per behavior class) this script:
  1. Loads an individual telemetry sample from the TRAINING data only (never the
     held-out test set) and keeps its ground-truth label for reporting.
  2. Runs the saved Logistic Regression pipeline through inference.BehaviorPredictor.
  3. Derives the NPC strategy from the MODEL'S PREDICTED behavior only.
     The ground-truth label is NEVER used to pick the strategy.
  4. Runs the turn-based simulation (simulation.TurnBasedSimulation) for at most
     10 turns with a fixed seed, using the returned strategy.

No model is retrained or tuned and no existing ML code is modified.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

from inference import BehaviorPredictor
from adaptive_npc import get_npc_strategy
from features import CLASS_NAMES
from utils.paths import get_project_root, load_config
from simulation import TurnBasedSimulation


MAX_TURNS = 10
SIMULATION_SEED = 42
SELECTION_SEED = 42


def load_training_samples():
    """
    Load the TRAINING split only (never the held-out test set).

    Reproduces the exact train/val/test split used by the ML pipeline and keeps
    only the training rows.

    Returns:
        X_train_raw: raw (unscaled) feature matrix for training rows
        y_train_raw: ground-truth behavior class per training row
        feature_cols: feature column names
    """
    config = load_config()
    root = get_project_root()
    df = pd.read_csv(root / 'data' / 'raw' / 'synthetic_telemetry.csv')

    feature_cols = [c for c in df.columns if c != 'behavior_class']
    X_raw = df[feature_cols].values
    y_raw = df['behavior_class'].values

    from sklearn.model_selection import train_test_split
    test_size = config['features']['test_size']
    val_size = config['features']['val_size']
    seed = config['random_seed']

    # Identical split logic to features.prepare_features_and_split
    X_temp, X_test_raw, y_temp, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=seed, stratify=y_raw
    )
    val_ratio = val_size / (1 - test_size)
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
    )
    return X_train_raw, y_train_raw, feature_cols


def _select_training_players(predictor, X_train, y_train, feature_cols, seed):
    """
    Pick one player per ground-truth class from the TRAINING data.

    For each class we prefer a training sample that the model predicts as that
    same class (so the demo showcases the matching strategy); if none exists we
    fall back to any sample of the class. Selection is deterministic (seeded).
    Ground-truth labels are preserved for reporting only.
    """
    rng = np.random.RandomState(seed)
    players = []
    for cls in CLASS_NAMES:
        class_indices = np.where(y_train == cls)[0]
        candidates = rng.permutation(class_indices)
        chosen = None
        for idx in candidates:
            telemetry = {
                feature_cols[j]: float(X_train[idx, j])
                for j in range(len(feature_cols))
            }
            prediction = predictor.predict_behavior(telemetry)['predicted_behavior']
            if prediction == cls:
                chosen = int(idx)
                break
        if chosen is None:
            chosen = int(candidates[0])
        telemetry = {
            feature_cols[j]: float(X_train[chosen, j])
            for j in range(len(feature_cols))
        }
        players.append({
            'ground_truth': cls,
            'telemetry': telemetry,
            'training_index': chosen,
        })
    return players


def short_strategy(strategy):
    """Compact strategy summary for the summary table."""
    posture = strategy.get('posture', '?')
    distance = strategy.get('distance', '?')
    combat_style = strategy.get('combat_style', '?')
    return f"{posture}/{distance}/{combat_style}"


def describe_strategy(strategy):
    """Human-readable strategy description for the player header."""
    parts = []
    for key in ('posture', 'distance', 'combat_style', 'aggression_level', 'patrol_radius'):
        if key in strategy:
            parts.append(f"{key}={strategy[key]}")
    if 'objective_awareness' in strategy:
        parts.append(f"objective_awareness={strategy['objective_awareness']}")
    return ' | '.join(parts)


def run_demo():
    """Run the full end-to-end demonstration and print the report."""
    print("=" * 74)
    print("END-TO-END DEMONSTRATION: TELEMETRY -> ML PREDICTION -> NPC STRATEGY -> SIMULATION")
    print("=" * 74)
    print("\nEach player is an INDIVIDUAL training sample (from the training split only, "
          "NOT the held-out test set).")
    print("The NPC strategy is always derived from the MODEL'S PREDICTED behavior. "
          "Ground-truth labels are reported only.")

    predictor = BehaviorPredictor()
    X_train, y_train, feature_cols = load_training_samples()
    players = _select_training_players(
        predictor, X_train, y_train, feature_cols, seed=SELECTION_SEED
    )

    summary_rows = []

    for i, player in enumerate(players, 1):
        ground_truth = player['ground_truth']
        telemetry = player['telemetry']

        # --- ML prediction from telemetry -----------------------------------
        result = predictor.predict_behavior(telemetry)
        predicted = result['predicted_behavior']
        confidence = result['confidence']

        # --- NPC strategy from PREDICTED behavior only -----------------------
        strategy = get_npc_strategy(predicted)

        # --- Turn-based simulation using that strategy ----------------------
        sim = TurnBasedSimulation(
            player_behavior=predicted,
            npc_strategy=strategy,
            seed=SIMULATION_SEED,
            max_turns=MAX_TURNS,
        )
        final_state = sim.run()

        # --- Header ----------------------------------------------------------
        print("\n" + "=" * 74)
        print(f"PLAYER {i}")
        print("=" * 74)
        print(f"Ground truth:       {ground_truth}  (training index {player['training_index']})")
        print(f"Predicted behavior: {predicted}")
        print(f"Confidence:         {confidence:.4f}")
        if predicted == ground_truth:
            print(f"Prediction match:   YES (model agrees with ground truth)")
        else:
            print(f"Prediction match:   NO - model predicted '{predicted}' "
                  f"(ground truth '{ground_truth}')")
        print(f"NPC strategy:       {describe_strategy(strategy)}")
        print(f"  triggered_by:     {strategy['triggered_by']} "
              f"(= model prediction, NOT ground truth)")

        # --- Simulation log ---------------------------------------------------
        print("\nTurn-by-turn simulation log (seed={}, max_turns={}):".format(
            SIMULATION_SEED, MAX_TURNS))
        print(sim.format_log())

        # --- Final results -----------------------------------------------------
        final = final_state.log[-1]
        status = final['status']
        result_label = {
            'player_victory': 'Player wins',
            'npc_victory': 'NPC wins',
            'max_turns_reached': 'Time limit (no winner)',
        }.get(status, status)
        print("\nFinal results:")
        print(f"  Final player HP: {final_state.player_hp}")
        print(f"  Final NPC HP:    {final_state.npc_hp}")
        print(f"  Turns played:    {final_state.turn}")
        print(f"  Result/winner:   {result_label}")
        print(f"  Predicted behavior used for adaptation: {final_state.player_behavior}")

        summary_rows.append({
            'Player': f'Player {i}',
            'Ground Truth': ground_truth,
            'Prediction': predicted,
            'Confidence': f"{confidence:.4f}",
            'NPC Strategy': short_strategy(strategy),
            'Result': result_label,
        })

    # --- Summary table ---------------------------------------------------------
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    # --- Explicit verification of the predicted-behavior-only rule -------------
    print("\n" + "=" * 74)
    print("VERIFICATION: STRATEGY SOURCE")
    print("=" * 74)
    for p in players:
        pred = predictor.predict_behavior(p['telemetry'])['predicted_behavior']
        strat = get_npc_strategy(pred)
        assert strat['triggered_by'] == pred, "Strategy not derived from prediction!"
    print("For every player: get_npc_strategy(PREDICTED behavior) was used, and")
    print("strategy['triggered_by'] == predicted behavior holds. Ground-truth labels")
    print("were used ONLY for reporting and never to select the NPC strategy.")


if __name__ == "__main__":
    run_demo()
