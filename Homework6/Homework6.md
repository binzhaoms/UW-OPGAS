Microsoft Introduction to AI and Machine Learning
Instructor: J. Nathan Kutz
HOMEWORK #6: REINFORCEMENT LEARNING
RL is often trained in what are called gym environments. Training gyms are simulation engines
which emulate either a true physical system environment, or the gym is a game (e.g. chess or go).
The gym allows an agent to interact with the virtual environment through simulation. Ultimately,
this allows the agent to learn an optimal policy π(S).
As a specific example for the use of RL, we develop a snake-apple game. The game is played on
a specified grid. The game can be cloned from the python repository from Vincent van Wynendaele:
https://github.com/Vinwcent/SnakeReinf
The snake starts with unit length (a worm which is the length of a single grid cell). The goal is for
the worm to learn to move towards the apple and eat it. The worm is rewarded and grows by one
unit length. The long term reward is for winning the game, which means that the worm has become
full length and occupies all cells. To structure the game play, the RL agent is allowed to take actions:
moving right, moving left, moving up and moving down.
Parameters of the game play are established before training, including the grid size and the
reward structure. The file launch.py contains the key element settings for the game. The following
code blocks specify how the game is played out
from game.Snake import Snake
from reinf.SnakeEnv import SnakeEnv
from reinf.utils import perform_mc, show_games
# Winning everytime hyperparameters
grid_length = 4 # box size
n_episodes = 10000 # number of trials run
epsilon = 0.1 # exploratory behavior (=0 no randomness, =1 totally random)
gamma = 0.98 # discount for future rewards
rewards = [-2, -1, 5, 1000] # see below
# [Losing move, inefficient move, efficient move, winning move]
The key settings are the parameter epsilon ∈ [0,1], which determines the degree of exploration, the
parameter gamma ∈ [0,1], which determines the discounted future reward, and the reward for a
losing move, inefficient move, efficient move and winning move. The winning move is when the snake
wins the game and becomes full length.
The game can be played interactively
game = Snake((800, 800), grid_length)
game.start_interactive_game()
Alternatively, one can run thousands, or hundreds of thousands of games in the gym environment
for training. Specifically, the games generate a policy optimization during game play
# Training part
env = SnakeEnv(grid_length=grid_length, with_rendering=False)
q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)
Once trained, the game can be played with the learned policy and visualized.
# Viz part (new games)
env = SnakeEnv(grid_length=grid_length, with_rendering=True)
show_games(env, 20, q_table)
In the above, 20 games are played using the trained q-table from the RL training module. By
tuning the parameters and playing enough games, one can teach the RL to win the game with high
probability. This is left as an exercise for this chapter.
NOTE: You will write a narrative report about this homework on your github page exploring the
tradeoffs between exploration, rewards and winning.