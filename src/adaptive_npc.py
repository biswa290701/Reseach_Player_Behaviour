"""
Adaptive NPC decision layer.

Maps predicted player behavior to NPC strategy parameters.
This is the adaptation logic that drives NPC AI behavior.
"""

from typing import Dict, Any
from features import CLASS_NAMES


# Strategy definitions for each predicted player behavior
NPC_STRATEGIES = {
    'Aggressive': {
        'posture': 'defensive',
        'distance': 'maintain_range',
        'combat_style': 'counterattack',
        'aggression_level': 0.3,
        'patrol_radius': 'normal',
        'description': 'Player is aggressive -> NPC plays defensively, maintains distance, counters attacks'
    },
    'Defensive': {
        'posture': 'offensive',
        'distance': 'close_gap',
        'combat_style': 'flanking',
        'aggression_level': 0.8,
        'patrol_radius': 'normal',
        'description': 'Player is defensive -> NPC increases pressure, closes distance, flanks'
    },
    'Explorer': {
        'posture': 'adaptive',
        'distance': 'dynamic',
        'combat_style': 'ambush_at_objectives',
        'aggression_level': 0.5,
        'patrol_radius': 'expanded',
        'objective_awareness': 'high',
        'description': 'Player explores -> NPC expands patrols, anticipates movement, ambushes at objectives'
    },
    'Balanced': {
        'posture': 'mixed',
        'distance': 'standard',
        'combat_style': 'standard',
        'aggression_level': 0.5,
        'patrol_radius': 'normal',
        'description': 'Player is balanced -> NPC uses standard mixed strategy'
    }
}


def get_npc_strategy(predicted_behavior: str) -> Dict[str, Any]:
    """
    Get NPC strategy parameters for a predicted player behavior.
    
    Args:
        predicted_behavior: One of 'Aggressive', 'Balanced', 'Defensive', 'Explorer'
        
    Returns:
        Dictionary of strategy parameters for the NPC AI
        
    Raises:
        ValueError: If behavior is not recognized
    """
    if predicted_behavior not in NPC_STRATEGIES:
        raise ValueError(
            f"Unknown behavior: '{predicted_behavior}'. "
            f"Expected one of: {list(NPC_STRATEGIES.keys())}"
        )
    
    # Return a copy to prevent accidental mutation
    strategy = NPC_STRATEGIES[predicted_behavior].copy()
    strategy['triggered_by'] = predicted_behavior
    return strategy


def get_all_strategies() -> Dict[str, Dict[str, Any]]:
    """Return all available NPC strategies (for inspection/debugging)."""
    return {k: v.copy() for k, v in NPC_STRATEGIES.items()}


def describe_strategy(strategy: Dict[str, Any]) -> str:
    """Generate a human-readable description of an NPC strategy."""
    behavior = strategy.get('triggered_by', 'Unknown')
    desc = strategy.get('description', 'No description')
    params = ', '.join(
        f"{k}={v}" for k, v in strategy.items() 
        if k not in ['description', 'triggered_by']
    )
    return f"[{behavior}] {desc} | Params: {params}"


if __name__ == "__main__":
    # Quick test
    for behavior in CLASS_NAMES:
        strategy = get_npc_strategy(behavior)
        print(describe_strategy(strategy))