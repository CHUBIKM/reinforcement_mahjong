#!/usr/bin/env python3
"""Debug entry point for the C++ Mahjong engine.

Usage:
    python debug_cpp_engine.py                  # Run full random game
    python debug_cpp_engine.py --seed 42        # Specific seed
    python debug_cpp_engine.py --step           # Step-by-step mode
    python debug_cpp_engine.py --obs            # Test observation encoding
    python debug_cpp_engine.py --parity 100     # Compare C++ vs Python (N games)
    python debug_cpp_engine.py --rl             # Test RL adapter integration
"""

import argparse
import sys
import time
import random

from mahjong.engine import RiichiEngine
from mahjong.rules import RuleProfile
from _mahjong_cpp import ActionType, Phase, OBS_DIM


def run_random_game(seed: int = 0, verbose: bool = True):
    """Run a single random game and print results."""
    config = RuleProfile()
    eng = RiichiEngine(seed=seed, config=config)
    eng.reset(dealer=0)

    if verbose:
        print(f"=== Random Game (seed={seed}) ===")
        for p in eng.players:
            from _mahjong_cpp import hand_to_str
            print(f"  Player {p.seat}: {hand_to_str(p.hand34)}")

    result = eng.play_random(verbose=verbose)

    print(f"\nResult: {result.reason}, done={result.done}")
    if result.winner is not None:
        print(f"  Winner: {result.winner}")
    if result.winners:
        print(f"  Winners: {result.winners}")
    if result.han > 0:
        print(f"  Han: {result.han}, Fu: {result.fu}")
    if result.yaku_list:
        print(f"  Yaku: {result.yaku_list}")
    if result.score_delta != [0, 0, 0, 0]:
        print(f"  Score delta: {result.score_delta}")
    if result.info:
        print(f"  Info: {result.info}")
    print(f"  Scores: {eng.scores}")
    return result


def step_by_step(seed: int = 0):
    """Interactive step-by-step game execution."""
    config = RuleProfile()
    eng = RiichiEngine(seed=seed, config=config)
    eng.reset(dealer=0)

    from _mahjong_cpp import hand_to_str, tile_to_str

    step_count = 0
    while not eng.done:
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        print(f"  Phase: {eng.phase}, Turn: {eng.turn}, Current: {eng.cur}")

        p = eng.players[eng.cur]
        print(f"  Player {eng.cur} hand: {hand_to_str(p.hand34)}")
        if p.river:
            river_str = " ".join(tile_to_str(t) for t in p.river)
            print(f"  River: {river_str}")

        actions = eng.legal_actions()
        print(f"  Legal actions ({len(actions)}): ", end="")
        for a in actions[:10]:
            print(f"{a}", end="  ")
        if len(actions) > 10:
            print(f"... +{len(actions)-10} more", end="")
        print()

        if eng.pending_discard:
            pd = eng.pending_discard
            print(f"  PendingDiscard: player={pd.get('player')}, tile={pd.get('tile')}, "
                  f"actor={pd.get('actor')}, claim_made={pd.get('claim_made', False)}")

        cmd = input("  [Enter]=auto, [q]=quit, [a <idx>]=apply action: ").strip()
        if cmd == "q":
            break
        elif cmd.startswith("a "):
            try:
                idx = int(cmd.split()[1])
                if 0 <= idx < len(actions):
                    result = eng.apply_action(actions[idx])
                    print(f"  => {result.reason}")
                else:
                    print(f"  Invalid index, must be 0..{len(actions)-1}")
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")
        else:
            # Auto-step: apply first legal action
            if eng.phase == "DRAW":
                eng.draw()
            elif actions:
                result = eng.apply_action(actions[0])
                print(f"  => {result.reason}")

    print(f"\nGame finished after {step_count} steps. Reason: {eng.done}")


def test_observation(seed: int = 0):
    """Test observation encoding (dict vs zero-copy array)."""
    import numpy as np

    config = RuleProfile()
    eng = RiichiEngine(seed=seed, config=config)
    eng.reset(dealer=0)
    eng.draw()

    # Dict observation
    obs_dict = eng.get_obs(seat=0)
    print("=== Dict Observation ===")
    for k, v in obs_dict.items():
        if isinstance(v, list) and len(v) > 10:
            print(f"  {k}: [{len(v)} items]")
        else:
            print(f"  {k}: {v}")

    # Zero-copy array observation
    obs_array = eng.get_obs_array(seat=0)
    print(f"\n=== Array Observation ===")
    print(f"  Shape: {obs_array.shape}, dtype: {obs_array.dtype}")
    print(f"  OBS_DIM: {OBS_DIM}")
    print(f"  First 34 values (hand34): {obs_array[:34].tolist()}")
    print(f"  Non-zero count: {np.count_nonzero(obs_array)}")

    # Verify consistency
    hand_from_dict = obs_dict["hand34"]
    hand_from_array = obs_array[:34].tolist()
    if hand_from_dict == hand_from_array:
        print("  hand34 MATCH between dict and array")
    else:
        print(f"  MISMATCH! dict={hand_from_dict}, array={hand_from_array}")


def test_parity(n_games: int = 100, seed_start: int = 0):
    """Compare C++ engine with pure Python engine for parity."""
    from mahjong._engine_pure import RiichiEngine as PyEngine
    from mahjong.rules import RuleProfile

    mismatches = 0
    start_time = time.time()

    for i in range(n_games):
        seed = seed_start + i
        config = RuleProfile()

        cpp_eng = RiichiEngine(seed=seed, config=config)
        cpp_eng.reset(dealer=0)
        cpp_result = cpp_eng.play_random(verbose=False)

        py_eng = PyEngine(seed=seed, config=config)
        py_eng.reset(dealer=0)
        py_result = py_eng.play_random(verbose=False)

        if cpp_result.reason != py_result.reason:
            print(f"  Seed {seed}: reason mismatch C++={cpp_result.reason} vs Py={py_result.reason}")
            mismatches += 1
        elif cpp_result.score_delta != py_result.score_delta:
            print(f"  Seed {seed}: score_delta mismatch C++={cpp_result.score_delta} vs Py={py_result.score_delta}")
            mismatches += 1

    elapsed = time.time() - start_time
    print(f"\nParity test: {n_games} games, {mismatches} mismatches, {elapsed:.2f}s")
    if mismatches == 0:
        print("All games match!")
    return mismatches


def test_rl_adapter():
    """Test RL adapter integration with C++ engine."""
    from mahjong.rl.adapter import OBS_DIM, N_ACTIONS, mask_builder, obs_encoder
    import numpy as np

    print(f"OBS_DIM={OBS_DIM}, N_ACTIONS={N_ACTIONS}")

    config = RuleProfile()
    eng = RiichiEngine(seed=42, config=config)
    eng.reset(dealer=0)

    # Test through several steps
    for step in range(20):
        if eng.done:
            break

        if eng.phase == "DRAW":
            eng.draw()

        actions = eng.legal_actions()
        if not actions:
            break

        # Build mask
        mask = mask_builder(eng, 0)
        assert mask.shape == (N_ACTIONS,), f"Mask shape: {mask.shape}"
        assert mask.sum() > 0, "No legal actions in mask"

        # Encode observation
        obs_vec = obs_encoder(eng, 0)
        assert obs_vec.shape == (OBS_DIM,), f"Obs shape: {obs_vec.shape}"

        # Apply random legal action
        action = random.choice(actions)
        result = eng.apply_action(action)

        if step < 5:
            print(f"  Step {step}: phase={eng.phase}, action={action}, "
                  f"mask_sum={mask.sum()}, obs_nonzero={np.count_nonzero(obs_vec)}")

    print(f"RL adapter test completed after {step+1} steps. Engine done={eng.done}")


def benchmark(n_games: int = 1000, seed_start: int = 0):
    """Benchmark C++ engine throughput."""
    config = RuleProfile()

    start = time.time()
    total_turns = 0
    for i in range(n_games):
        eng = RiichiEngine(seed=seed_start + i, config=config)
        eng.logging_enabled = False  # Skip logging for max speed
        eng.reset(dealer=0)
        result = eng.play_random()
        total_turns += eng.turn
    elapsed = time.time() - start

    games_per_sec = n_games / elapsed
    turns_per_sec = total_turns / elapsed
    print(f"Benchmark: {n_games} games in {elapsed:.2f}s")
    print(f"  {games_per_sec:.0f} games/s, {turns_per_sec:.0f} turns/s")
    print(f"  Avg turns/game: {total_turns/n_games:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug C++ Mahjong engine")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--step", action="store_true", help="Step-by-step mode")
    parser.add_argument("--obs", action="store_true", help="Test observation encoding")
    parser.add_argument("--parity", type=int, metavar="N", help="Compare C++ vs Python (N games)")
    parser.add_argument("--rl", action="store_true", help="Test RL adapter integration")
    parser.add_argument("--bench", type=int, metavar="N", help="Benchmark N games")
    args = parser.parse_args()

    try:
        if args.step:
            step_by_step(seed=args.seed)
        elif args.obs:
            test_observation(seed=args.seed)
        elif args.parity:
            test_parity(n_games=args.parity, seed_start=args.seed)
        elif args.rl:
            test_rl_adapter()
        elif args.bench:
            benchmark(n_games=args.bench, seed_start=args.seed)
        else:
            run_random_game(seed=args.seed)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the C++ extension is built: pip install -e .")
        sys.exit(1)
