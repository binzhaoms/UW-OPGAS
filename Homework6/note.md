# Homework 6 Notes

## Scope
Reinforcement learning on Snake environment using the SnakeReinf repository.

## Step 1: Dependency fetched and inspected
- Cloned SnakeReinf into `Homework6/SnakeReinf`.
- Verified required modules from assignment prompt are present:
	- `game/Snake.py`
	- `reinf/SnakeEnv.py`
	- `reinf/utils.py`
- Verified required symbols exist:
	- `class Snake`
	- `class SnakeEnv`
	- `def perform_mc`

## Step 2: Environment setup complete
- Configured project venv at `/Users/binzhaoms/Dev/UW-OPGAS/.venv`.
- Installed Python dependencies:
	- `gymnasium`
	- `tqdm`
	- `pygame`
- Resolved macOS build issue for pygame (`SDL.h` missing) by installing system packages with Homebrew:
	- `sdl2`, `sdl2_image`, `sdl2_mixer`, `sdl2_ttf`, `pkg-config`
- Runtime validation passed:
	- `Snake`, `SnakeEnv`, `perform_mc`, `show_games` import correctly
	- environment can be instantiated with `SnakeEnv(grid_length=4, with_rendering=False)`

## Step 3: Training script implemented
- Added script: `Homework6/train_snake_rl.py`
- Features:
	- configurable RL hyperparameters (`grid_length`, `episodes`, `epsilon`, `gamma`, `rewards`, `seed`, `run_name`)
	- training via `perform_mc`
	- evaluation over N games with greedy policy
	- saves outputs in `Homework6/results`
- Syntax check passed (`py_compile`).

## Step 4: Initial baseline run and validation
- Baseline command:
	- run name: `baseline`
	- grid length: `4`
	- episodes: `2000`
	- epsilon: `0.1`
	- gamma: `0.98`
	- rewards: `[-2, -1, 5, 1000]`
	- eval games: `100`
	- seed: `42`
- Found and fixed one issue:
	- pickling `defaultdict(lambda: ...)` failed
	- fixed by serializing `dict(q_table)` in `train_snake_rl.py`

## Baseline results
- training seconds: `0.0975`
- states in q-table: `1408`
- wins/losses: `0/100`
- win rate: `0.0`
- average reward: `0.11`
- average steps: `3.99`

## Generated files
- `Homework6/results/baseline_q_table.pkl`
- `Homework6/results/baseline_summary.json`
- `Homework6/results/latest_summary.json`

## Next step
Step 6: write Homework 6 report from the baseline + tuning comparisons.

## Step 5: Tuning parameters and comparing results
All tuning runs used:
- `grid_length=4`
- `episodes=10000`
- `eval_games=200`
- `seed=42`

### Experiments run
- `tune_ep10000`: epsilon=0.1, gamma=0.98, rewards=[-2, -1, 5, 1000]
- `tune_eps005`: epsilon=0.05, gamma=0.98, rewards=[-2, -1, 5, 1000]
- `tune_eps020`: epsilon=0.2, gamma=0.98, rewards=[-2, -1, 5, 1000]
- `tune_gam090`: epsilon=0.1, gamma=0.9, rewards=[-2, -1, 5, 1000]
- `tune_rew_small`: epsilon=0.1, gamma=0.98, rewards=[-1, -0.2, 1, 50]

### Comparison summary
- `baseline`:
	- win_rate=0.000
	- avg_reward=0.110
	- avg_steps=3.990
	- q_states=1408

- `tune_ep10000`:
	- win_rate=0.000
	- avg_reward=0.490
	- avg_steps=4.705
	- q_states=3688

- `tune_eps005`:
	- win_rate=0.000
	- avg_reward=0.415
	- avg_steps=4.775
	- q_states=3184

- `tune_eps020`:
	- win_rate=0.000
	- avg_reward=0.815
	- avg_steps=6.830
	- q_states=4011

- `tune_gam090`:
	- win_rate=0.000
	- avg_reward=0.605
	- avg_steps=4.650
	- q_states=3918

- `tune_rew_small`:
	- win_rate=0.000
	- avg_reward=0.305
	- avg_steps=3.940
	- q_states=3237

### Findings
- Increasing episode count from 2000 to 10000 improved value estimates and average reward, but still did not produce wins on this setup.
- Larger exploration (`epsilon=0.2`) increased average reward and trajectory length, suggesting broader state coverage.
- Lower discount (`gamma=0.9`) improved short-horizon average reward relative to baseline epsilon/gamma at 10000 episodes.
- Compressing reward magnitudes reduced average reward signal and did not improve success rate.
- Across these runs on 4x4 grid, no configuration reached non-zero win rate, so additional tuning is still needed.

### Files generated for tuning
- `Homework6/results/tune_ep10000_q_table.pkl`
- `Homework6/results/tune_ep10000_summary.json`
- `Homework6/results/tune_eps005_q_table.pkl`
- `Homework6/results/tune_eps005_summary.json`
- `Homework6/results/tune_eps020_q_table.pkl`
- `Homework6/results/tune_eps020_summary.json`
- `Homework6/results/tune_gam090_q_table.pkl`
- `Homework6/results/tune_gam090_summary.json`
- `Homework6/results/tune_rew_small_q_table.pkl`
- `Homework6/results/tune_rew_small_summary.json`
