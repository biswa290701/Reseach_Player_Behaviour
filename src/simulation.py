"""
Turn-based NPC simulation engine.

Simulates combat between a scripted player (driven by a behavior class) and an
NPC (driven by the strategy dictionary produced by adaptive_npc.get_npc_strategy).

This is a research prototype, NOT a real game engine. It intentionally has no
reinforcement learning, no telemetry features, and no dependency on the trained
classifier. The simulation only consumes the strategy *parameters*.

Design goals:
- Deterministic: identical (behavior, strategy, seed) inputs produce identical logs.
- Independent: nothing here imports inference.py, models.py or the trained pipeline.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Tunable simulation constants
# ---------------------------------------------------------------------------
PLAYER_HP = 100
NPC_HP = 100
PLAYER_ATTACK_DAMAGE = 20
NPC_ATTACK_DAMAGE = 18
DODGE_CHANCE = 0.5
RISKY_ATTACK_MISS_CHANCE = 0.2
COUNTERATTACK_CHANCE = 0.5
FLANK_MULTIPLIER = 1.5
AMBUSH_MULTIPLIER = 1.5
DEFAULT_MAX_TURNS = 30

# Ordered proximity ladder used to move between distance bands.
DISTANCE_ORDER = ['close', 'medium', 'far']
DISTANCE_INDEX = {'close': 0, 'medium': 1, 'far': 2}

# Player action pool per predicted behavior.
PLAYER_ACTIONS_BY_BEHAVIOR: Dict[str, List[str]] = {
    'Aggressive': ['attack', 'charge', 'risky_attack'],
    'Balanced': ['attack', 'defend', 'reposition'],
    'Defensive': ['defend', 'retreat', 'block'],
    'Explorer': ['move_to_objective', 'scout', 'ambush'],
}

# Weighting per behavior so each archetype feels distinct.
PLAYER_ACTION_WEIGHTS: Dict[str, List[float]] = {
    'Aggressive': [0.5, 0.3, 0.2],
    'Balanced': [0.5, 0.3, 0.2],
    'Defensive': [0.4, 0.3, 0.3],
    'Explorer': [0.5, 0.3, 0.2],
}

PLAYER_ATTACK_ACTIONS = {'attack', 'charge', 'risky_attack', 'ambush'}
NPC_ATTACK_ACTIONS = {'attack', 'charge', 'flank', 'counterattack', 'ambush'}


def _shift_distance(distance: str, delta: int) -> str:
    """Move one band up/down the proximity ladder, clamped at the ends."""
    idx = max(0, min(len(DISTANCE_ORDER) - 1, DISTANCE_INDEX[distance] + delta))
    return DISTANCE_ORDER[idx]


def _damage_vs_defense(dmg: int, defense_action: str, rng: random.Random) -> int:
    """
    Resolve a strike against the defender's action.

    Returns the actual damage applied.
    - block -> 0
    - dodge -> 0 with DODGE_CHANCE, else full
    - defend -> half damage
    - retreat / anything else -> full damage
    """
    if defense_action == 'block':
        return 0
    if defense_action == 'dodge':
        if rng.random() < DODGE_CHANCE:
            return 0
        return dmg
    if defense_action == 'defend':
        return int(dmg * 0.5)
    return dmg


@dataclass
class SimulationState:
    """
    Snapshot of the combat simulation.

    Fields:
        turn: Current turn number (0 before the first turn resolves).
        player_hp / npc_hp: Current health values.
        distance: One of 'close', 'medium', 'far'.
        player_action / npc_action: Actions chosen for the current turn.
        player_behavior: Behavior class driving the scripted player.
        npc_strategy: Strategy dict produced by adaptive_npc.get_npc_strategy().
        log: List of per-turn event entries.
        player_at_objective: Whether the player is currently camped on an objective
            (used by the ambush_at_objectives combat style).
    """

    turn: int = 0
    player_hp: int = PLAYER_HP
    npc_hp: int = NPC_HP
    distance: str = 'medium'
    player_action: Optional[str] = None
    npc_action: Optional[str] = None
    player_behavior: Optional[str] = None
    npc_strategy: Optional[Dict[str, Any]] = None
    log: List[Dict[str, Any]] = field(default_factory=list)
    player_at_objective: bool = False


class TurnBasedSimulation:
    """Drives a turn-based combat encounter between the scripted player and the NPC."""

    def __init__(
        self,
        player_behavior: str,
        npc_strategy: Dict[str, Any],
        seed: int = 42,
        max_turns: int = DEFAULT_MAX_TURNS,
        start_distance: str = 'medium',
    ) -> None:
        if player_behavior not in PLAYER_ACTIONS_BY_BEHAVIOR:
            raise ValueError(
                f"Unknown player behavior: '{player_behavior}'. "
                f"Expected one of: {list(PLAYER_ACTIONS_BY_BEHAVIOR.keys())}"
            )
        if not npc_strategy:
            raise ValueError("npc_strategy must be a non-empty dict (see adaptive_npc.get_npc_strategy)")
        if start_distance not in DISTANCE_INDEX:
            raise ValueError(f"start_distance must be one of {DISTANCE_ORDER}")

        self.seed = seed
        self.max_turns = max_turns
        # Seeded RNG keeps every run reproducible for a given seed.
        self.rng = random.Random(seed)
        self.state = SimulationState(
            player_behavior=player_behavior,
            npc_strategy=dict(npc_strategy),
            distance=start_distance,
        )

    # ------------------------------------------------------------------
    # Player decision logic
    # ------------------------------------------------------------------
    def _choose_player_action(self) -> str:
        actions = PLAYER_ACTIONS_BY_BEHAVIOR[self.state.player_behavior]
        weights = PLAYER_ACTION_WEIGHTS[self.state.player_behavior]
        return self.rng.choices(actions, weights=weights)[0]

    # ------------------------------------------------------------------
    # NPC decision logic (driven entirely by the strategy dict)
    # ------------------------------------------------------------------
    def _npc_decide_action(self) -> str:
        strategy = self.state.npc_strategy
        posture = strategy.get('posture', 'mixed')
        distance_pref = strategy.get('distance', 'standard')
        combat_style = strategy.get('combat_style', 'standard')
        rng = self.rng
        distance = self.state.distance
        player_action = self.state.player_action
        at_objective = self.state.player_at_objective

        # counterattack: bait the attacking player into a block -> riposte
        if combat_style == 'counterattack' and player_action in PLAYER_ATTACK_ACTIONS:
            return 'block' if rng.random() < 0.6 else 'dodge'

        # maintain_range: back off when the fight gets too close
        if distance_pref == 'maintain_range' and distance == 'close' and rng.random() < 0.5:
            return 'reposition'

        # close_gap: rush in when the fight is not yet at close range
        if distance_pref == 'close_gap' and distance != 'close' and rng.random() < 0.5:
            return 'charge'

        if posture == 'defensive':
            return rng.choices(('block', 'dodge', 'counterattack'), weights=(0.5, 0.3, 0.2))[0]

        if posture == 'offensive':
            if distance == 'far':
                return 'charge'
            if combat_style == 'flanking' and rng.random() < 0.6:
                return 'flank'
            return 'attack'

        if posture == 'adaptive':
            if at_objective and combat_style == 'ambush_at_objectives':
                return 'ambush'
            if distance == 'far':
                return 'charge'
            if distance == 'close':
                return rng.choices(('flank', 'attack'), weights=(0.5, 0.5))[0]
            return 'attack'

        # mixed / standard fallback
        return rng.choices(('attack', 'block', 'dodge', 'flank'), weights=(0.5, 0.25, 0.15, 0.1))[0]

    # ------------------------------------------------------------------
    # Combat resolution
    # ------------------------------------------------------------------
    def _resolve_turn(self) -> Dict[str, Any]:
        state = self.state
        rng = self.rng
        combat_style = state.npc_strategy.get('combat_style', 'standard')
        events: List[str] = []
        player_damage = 0
        npc_damage = 0
        distance = state.distance

        # ---- Player strikes ----
        if state.player_action in PLAYER_ATTACK_ACTIONS:
            if state.player_action == 'charge':
                distance = _shift_distance(distance, -1)
                events.append("Player charges in.")

            if state.player_action == 'ambush':
                if distance == 'far':
                    distance = _shift_distance(distance, -1)
                    events.append("Player closes to medium range before ambushing.")
                else:
                    dmg = int(PLAYER_ATTACK_DAMAGE * AMBUSH_MULTIPLIER)
                    dealt = _damage_vs_defense(dmg, state.npc_action, rng)
                    player_damage += dealt
                    events.append(f"Player ambush dealt {dealt} damage.")

            elif state.player_action == 'risky_attack':
                if rng.random() < RISKY_ATTACK_MISS_CHANCE:
                    events.append("Player's risky attack misses!")
                else:
                    dmg = int(PLAYER_ATTACK_DAMAGE * 1.5)
                    dealt = _damage_vs_defense(dmg, state.npc_action, rng)
                    player_damage += dealt
                    events.append(f"Player risky attack dealt {dealt} damage.")

            elif state.player_action == 'attack':
                if distance == 'far':
                    events.append("Player is out of range and cannot attack.")
                else:
                    dealt = _damage_vs_defense(PLAYER_ATTACK_DAMAGE, state.npc_action, rng)
                    player_damage += dealt
                    events.append(f"Player attack dealt {dealt} damage.")

            # Counterattack riposte after a blocked player attack.
            if state.npc_action == 'block' and combat_style == 'counterattack':
                if rng.random() < COUNTERATTACK_CHANCE:
                    riposte = NPC_ATTACK_DAMAGE
                    npc_damage += riposte
                    events.append(f"NPC counterattacks for {riposte} damage after blocking!")

        # ---- NPC strikes ----
        if state.npc_action in NPC_ATTACK_ACTIONS:
            base = NPC_ATTACK_DAMAGE
            if state.npc_action == 'flank':
                base = int(base * FLANK_MULTIPLIER)
            elif state.npc_action == 'ambush':
                base = int(base * AMBUSH_MULTIPLIER)

            if state.npc_action == 'charge':
                distance = _shift_distance(distance, -1)
            elif state.npc_action in ('attack', 'flank', 'ambush') and distance == 'far':
                events.append("NPC attack misses - out of range.")
            else:
                dealt = _damage_vs_defense(base, state.player_action, rng)
                npc_damage += dealt
                events.append(f"NPC {state.npc_action} dealt {dealt} damage.")

        # ---- Movement from non-combat actions ----
        if state.player_action in ('retreat', 'reposition'):
            distance = _shift_distance(distance, +1)
        if state.npc_action == 'reposition':
            distance = _shift_distance(distance, +1)

        # ---- Apply damage and update objective awareness ----
        state.player_hp = max(0, state.player_hp - npc_damage)
        state.npc_hp = max(0, state.npc_hp - player_damage)
        state.distance = distance

        if state.player_action == 'move_to_objective':
            state.player_at_objective = True
        elif state.player_action in ('scout', 'retreat', 'reposition'):
            state.player_at_objective = False

        return {
            'turn': state.turn,
            'distance': state.distance,
            'player_action': state.player_action,
            'npc_action': state.npc_action,
            'events': events,
            'player_hp': state.player_hp,
            'npc_hp': state.npc_hp,
            'status': 'ongoing',
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> SimulationState:
        """Run the encounter until a side reaches 0 HP or max_turns is hit."""
        state = self.state
        while state.turn < self.max_turns and state.player_hp > 0 and state.npc_hp > 0:
            state.turn += 1
            state.player_action = self._choose_player_action()
            state.npc_action = self._npc_decide_action()
            state.log.append(self._resolve_turn())

        last = state.log[-1]
        if state.npc_hp <= 0:
            last['status'] = 'player_victory'
        elif state.player_hp <= 0:
            last['status'] = 'npc_victory'
        else:
            last['status'] = 'max_turns_reached'

        state.log.append({
            'turn': state.turn,
            'status': last['status'],
            'player_hp': state.player_hp,
            'npc_hp': state.npc_hp,
        })
        return state

    # ------------------------------------------------------------------
    # Log formatting
    # ------------------------------------------------------------------
    def format_log(self) -> str:
        """Render the event log as readable turn-by-turn lines."""
        lines = []
        header = (
            f"Simulation | behavior={self.state.player_behavior} "
            f"| seed={self.seed} | max_turns={self.max_turns}"
        )
        lines.append(header)
        strategy = self.state.npc_strategy or {}
        lines.append(
            f"NPC strategy | posture={strategy.get('posture')} "
            f"| distance={strategy.get('distance')} "
            f"| combat_style={strategy.get('combat_style')}"
        )
        for entry in self.state.log:
            if 'events' not in entry:
                continue
            lines.append(
                f"Turn {entry['turn']:>2} | dist={entry['distance']:<6} "
                f"| player={entry['player_action']:<16} "
                f"| npc={entry['npc_action']:<14} "
                f"| HP {entry['player_hp']:>3}/{entry['npc_hp']:<3}"
            )
            for event in entry['events']:
                lines.append(f"    - {event}")
        if self.state.log:
            final = self.state.log[-1]
            lines.append(f"Outcome: {final.get('status')}")
        return '\n'.join(lines)


def run_simulation(
    player_behavior: str,
    npc_strategy: Dict[str, Any],
    seed: int = 42,
    max_turns: int = DEFAULT_MAX_TURNS,
    start_distance: str = 'medium',
) -> SimulationState:
    """Convenience wrapper that runs a single encounter and returns the final state."""
    sim = TurnBasedSimulation(
        player_behavior=player_behavior,
        npc_strategy=npc_strategy,
        seed=seed,
        max_turns=max_turns,
        start_distance=start_distance,
    )
    return sim.run()
