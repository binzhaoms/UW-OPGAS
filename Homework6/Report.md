# Homework 6 Report: Reinforcement Learning on Snake

## 1. Objective
The objective of this homework was to train a reinforcement learning (RL) agent in the Snake environment and explore the tradeoffs between:
- exploration (epsilon),
- discounted future reward (gamma),
- reward shaping,
- and winning performance.

The required environment came from the SnakeReinf repository and uses a Monte Carlo control method to learn a q-table policy.

---

## 2. What We Did in Each Step

### Step 1: Fetch and inspect dependency
What we did:
- Cloned SnakeReinf into Homework6/SnakeReinf.
- Verified required modules from the assignment:
  - game/Snake.py
  - reinf/SnakeEnv.py
  - reinf/utils.py
- Verified required classes/functions are available:
  - Snake
  - SnakeEnv
  - perform_mc

What we found:
- The repository structure matched the homework prompt exactly.
- The provided API is sufficient to run training and visualization directly.

### Step 2: Set up runtime environment
What we did:
- Configured and used the project venv.
- Installed Python dependencies listed by the repository:
  - gymnasium
  - pygame
  - tqdm
- Fixed pygame installation issue on macOS by installing SDL system libraries:
  - sdl2, sdl2_image, sdl2_mixer, sdl2_ttf, pkg-config
- Validated imports and environment construction with SnakeEnv(grid_length=4, with_rendering=False).

What we found:
- Python packages alone were not enough on macOS; SDL headers were required.
- After SDL install, all imports and environment initialization worked correctly.

### Step 3: Implement training pipeline
What we did:
- Implemented Homework6/train_snake_rl.py.
- Added CLI parameters for:
  - grid_length
  - episodes
  - epsilon
  - gamma
  - rewards (lose, inefficient, efficient, win)
  - eval_games
  - seed
  - run_name
- Added post-training greedy evaluation and artifact export to Homework6/results.
- Saved:
  - q-table as pickle
  - run summary as json
  - latest summary snapshot

What we found:
- The script made experiment iteration reproducible and fast.
- One serialization issue appeared later with defaultdict(lambda), which was fixed in step 4.

### Step 4: Run baseline and validate
What we did:
- Ran baseline experiment with:
  - grid_length=4
  - episodes=2000
  - epsilon=0.1
  - gamma=0.98
  - rewards=[-2, -1, 5, 1000]
  - eval_games=100
- Fixed a pickling error by converting q_table to plain dict before serialization.

What we found:
- Baseline training completed quickly and generated all expected files.
- Baseline result:
  - win_rate = 0.000 (0/100)
  - avg_reward = 0.110
  - avg_steps = 3.990
  - q-table states = 1408

### Step 5: Tune and compare
What we did:
- Increased episode count and varied key hyperparameters:
  - epsilon (0.05, 0.1, 0.2)
  - gamma (0.9 vs 0.98)
  - reward scale ([ -2, -1, 5, 1000 ] vs [ -1, -0.2, 1, 50 ])
- Kept grid_length=4 and eval_games=200 for comparison consistency.
- Logged all run summaries in Homework6/results.

What we found:
- No tested configuration reached non-zero win rate.
- However, policy quality indicators improved:
  - average reward increased significantly over baseline
  - explored state count increased
  - trajectory length changed with exploration settings

---

## 3. Algorithm and Parameter Choices: What and Why

### Algorithm chosen
We used first-visit Monte Carlo control with epsilon-greedy action selection, as provided by perform_mc in reinf/utils.py.

Why this algorithm:
- It is exactly aligned with the provided homework environment implementation.
- It directly optimizes action values from episodic returns.
- It is simple and interpretable for studying exploration and reward tradeoffs.

### Core parameters chosen
- grid_length = 4
  - Reason: matches assignment example and keeps state/action space manageable.
- episodes = 2000 baseline, then 10000 tuning
  - Reason: baseline first for fast validation; higher episodes to improve value estimates.
- epsilon in {0.05, 0.1, 0.2}
  - Reason: compare low, medium, and higher exploration rates.
- gamma in {0.9, 0.98}
  - Reason: compare shorter-horizon versus longer-horizon return emphasis.
- reward sets:
  - default: [-2, -1, 5, 1000]
  - compressed: [-1, -0.2, 1, 50]
  - Reason: evaluate how reward magnitude affects learning signal and behavior.

---

## 4. How We Tuned and Reached the Result

### Tuning protocol
1. Establish baseline run and verify full pipeline.
2. Increase training episodes to improve policy estimation.
3. Sweep epsilon to evaluate exploration impact.
4. Sweep gamma to evaluate future reward sensitivity.
5. Modify reward scale to evaluate shaping impact.
6. Compare by:
   - win_rate
   - avg_reward
   - avg_steps
   - number of states in q-table

### Experiment results
- baseline
  - episodes=2000, epsilon=0.1, gamma=0.98, rewards=[-2,-1,5,1000]
  - win_rate=0.000, avg_reward=0.110, avg_steps=3.990, states=1408

- tune_ep10000
  - episodes=10000, epsilon=0.1, gamma=0.98, rewards=[-2,-1,5,1000]
  - win_rate=0.000, avg_reward=0.490, avg_steps=4.705, states=3688

- tune_eps005
  - episodes=10000, epsilon=0.05, gamma=0.98, rewards=[-2,-1,5,1000]
  - win_rate=0.000, avg_reward=0.415, avg_steps=4.775, states=3184

- tune_eps020
  - episodes=10000, epsilon=0.2, gamma=0.98, rewards=[-2,-1,5,1000]
  - win_rate=0.000, avg_reward=0.815, avg_steps=6.830, states=4011

- tune_gam090
  - episodes=10000, epsilon=0.1, gamma=0.9, rewards=[-2,-1,5,1000]
  - win_rate=0.000, avg_reward=0.605, avg_steps=4.650, states=3918

- tune_rew_small
  - episodes=10000, epsilon=0.1, gamma=0.98, rewards=[-1,-0.2,1,50]
  - win_rate=0.000, avg_reward=0.305, avg_steps=3.940, states=3237

### Final interpretation
- The best setting in this sweep by average reward was epsilon=0.2 with 10000 episodes.
- Increasing episodes and exploration improved learned value coverage and reward behavior.
- Reward compression weakened learning signal relative to the default reward scale.
- Despite better intermediate metrics, none of these settings reached a winning policy yet on the evaluated runs.

---

## 5. Deliverables Generated
- Homework6/results/baseline_q_table.pkl
- Homework6/results/baseline_summary.json
- Homework6/results/tune_ep10000_q_table.pkl
- Homework6/results/tune_ep10000_summary.json
- Homework6/results/tune_eps005_q_table.pkl
- Homework6/results/tune_eps005_summary.json
- Homework6/results/tune_eps020_q_table.pkl
- Homework6/results/tune_eps020_summary.json
- Homework6/results/tune_gam090_q_table.pkl
- Homework6/results/tune_gam090_summary.json
- Homework6/results/tune_rew_small_q_table.pkl
- Homework6/results/tune_rew_small_summary.json
- Homework6/results/latest_summary.json

---

## 6. Next Tuning Ideas
To push from improved rewards to actual wins, likely next actions are:
- run much longer training (for example 50k to 200k episodes),
- test epsilon decay instead of fixed epsilon,
- test stronger terminal win reward and stronger inefficient-step penalty,
- evaluate multiple random seeds and average results.
