# Homework 6 Plan

## Goal
Implement Homework 6 using the Snake reinforcement learning environment and tune parameters to maximize winning probability.

## Step-by-step plan

### Step 1: Fetch / inspect SnakeReinf dependency
- Determine whether the `SnakeReinf` environment is already present locally.
- If not, clone or download `https://github.com/Vinwcent/SnakeReinf` into the repository.
- Confirm the key modules:
  - `game/Snake.py`
  - `reinf/SnakeEnv.py`
  - `reinf/utils.py`
- Verify imports for `Snake`, `SnakeEnv`, `perform_mc`, and `show_games`.

### Step 2: Set up the environment
- Install required Python packages for gym-style RL, if needed.
- Confirm the current virtual environment can run the Snake environment and training code.
- If necessary, add dependencies to the repo's instructions or a requirements file.

### Step 3: Implement the RL training pipeline
- Create a training script for Homework 6.
- Set the key parameters:
  - `grid_length`
  - `n_episodes`
  - `epsilon`
  - `gamma`
  - `rewards = [losing, inefficient, efficient, winning]`
- Use:
  - `env = SnakeEnv(grid_length=grid_length, with_rendering=False)`
  - `q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)`
- Save the learned `q_table` and any training statistics.

### Step 4: Run an initial baseline experiment
- Train the agent with a default parameter set.
- Verify training completes successfully.
- Run a small number of test games with rendering to confirm the learned policy works.

### Step 5: Tune parameters and compare results
- Explore how different settings affect learning and success:
  - `epsilon` values for exploration/exploitation balance
  - `gamma` values for future reward discounting
  - reward weights for losing, inefficient, efficient, and winning moves
  - optional grid size changes
- Record win rates or success probabilities for each configuration.
- Identify the tradeoffs between exploration, reward shaping, and winning performance.

### Step 6: Write the Homework 6 report
- Document the RL setup and implementation approach.
- Explain the tuning process and parameter choices.
- Present results from the baseline and tuned experiments.
- Discuss the behavior of the learned policy and the exploration/reward tradeoffs.
- Add the narrative to the GitHub report page.
