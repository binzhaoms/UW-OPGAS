import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def _prepare_imports(homework6_dir: Path):
    snake_repo = homework6_dir / "SnakeReinf"
    if not snake_repo.exists():
        raise FileNotFoundError(
            f"SnakeReinf not found at {snake_repo}. Run step 1 first."
        )
    sys.path.insert(0, str(snake_repo))

    from reinf.SnakeEnv import SnakeEnv  # type: ignore
    from reinf.utils import perform_mc  # type: ignore

    return SnakeEnv, perform_mc


def choose_greedy_action(state, valid_actions, q_table):
    state_key = tuple(tuple(x) for x in state)
    q_values = q_table[state_key]
    best_action = max(valid_actions, key=lambda a: q_values[a])
    return int(best_action)


def evaluate_policy(SnakeEnv, q_table, grid_length: int, n_games: int, seed: int):
    np.random.seed(seed)
    env = SnakeEnv(grid_length=grid_length, with_rendering=False)

    wins = 0
    losses = 0
    total_reward = 0.0
    steps_per_game = []

    for _ in range(n_games):
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0.0

        while not done and steps < 300:
            valid_actions = env.get_valid_actions(state)
            action = choose_greedy_action(state, valid_actions, q_table)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                if reward == 10:
                    wins += 1
                else:
                    losses += 1

        if not done:
            losses += 1

        total_reward += episode_reward
        steps_per_game.append(steps)

    win_rate = wins / n_games if n_games > 0 else 0.0
    avg_reward = total_reward / n_games if n_games > 0 else 0.0
    avg_steps = float(np.mean(steps_per_game)) if steps_per_game else 0.0

    return {
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Homework 6 Snake RL training")
    parser.add_argument("--grid-length", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--rewards", nargs=4, type=float, default=[-2, -1, 5, 1000])
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="baseline")
    args = parser.parse_args()

    homework6_dir = Path(__file__).resolve().parent
    results_dir = homework6_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    SnakeEnv, perform_mc = _prepare_imports(homework6_dir)

    np.random.seed(args.seed)

    train_env = SnakeEnv(grid_length=args.grid_length, with_rendering=False)

    start = time.time()
    q_table = perform_mc(
        env=train_env,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        gamma=args.gamma,
        rewards=args.rewards,
    )
    train_time = time.time() - start

    eval_summary = evaluate_policy(
        SnakeEnv=SnakeEnv,
        q_table=q_table,
        grid_length=args.grid_length,
        n_games=args.eval_games,
        seed=args.seed,
    )

    # Convert defaultdict(lambda: ...) to plain dict for safe pickling.
    q_table_plain = dict(q_table)
    q_table_path = results_dir / f"{args.run_name}_q_table.pkl"
    with q_table_path.open("wb") as f:
        pickle.dump(q_table_plain, f)

    summary = {
        "run_name": args.run_name,
        "grid_length": args.grid_length,
        "episodes": args.episodes,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "rewards": list(args.rewards),
        "seed": args.seed,
        "training_seconds": train_time,
        "num_states_in_q_table": len(q_table),
        "evaluation": eval_summary,
        "q_table_file": str(q_table_path.relative_to(homework6_dir.parent)),
    }

    summary_path = results_dir / f"{args.run_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    latest_path = results_dir / "latest_summary.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training completed.")
    print(json.dumps(summary, indent=2))
    print(f"Saved q_table to: {q_table_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
