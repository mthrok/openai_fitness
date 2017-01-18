"""Tutorial for RL introduction, following Chapter 3 of Sutton"""
# <markdowncell>

# # Chapter 3. Reinforcement Learning Problem
# In this tutorial we go through the example of computing action-value
# function and state-value function following the example given in Chapter 3
# of `Reinforcement Learning: An Introduction`.

# <markdowncell>

# State value function and action value function for policy $\pi$ are defined
# as follow

# \begin{align*}
# v_{\pi} (s)
#   &= \mathbb{E}_{\pi} \lbrack G_t | S_t = s \rbrack \\
# q_{\pi}(s, a)
#   &= \mathbb{E}_{\pi} \lbrack G_t | S_t = s, A_t = a \rbrack \\
# \text{where } & \text{$G_t$ , the sum of the discounted rewards, is} \\
# G_t
#   &= R_{t+1} + R_{t+2} + R_{t+3} + \dots + R_{T}
# \end{align*}

# In case of Markov Decision Process, the following can be derived.

# \begin{align*}
# v_{\pi} (s)
#   &= \mathbb{E}_{\pi} \lbrack G_t | S_t = s \rbrack \\
#   &= \mathbb{E}_{\pi} \lbrack \sum_{k=0}^{\infty}
#      \gamma ^ {k} R_{t+k+1} | S_t = s \rbrack \\
# q_{\pi}(s, a)
#   &= \mathbb{E}_{\pi} \lbrack G_t | S_t = s, A_t = a \rbrack \\
#   &= \mathbb{E}_{\pi} \lbrack \sum_{k=0}^{\infty}
#      \gamma ^ {k} R_{t+k+1} | S_t = s, A_t = a \rbrack
# \end{align*}

# These interweived functions statisfy recursive relationships as follow

# \begin{align*}
# v_{\pi} (s)
#   &= \mathbb{E}_{\pi} \lbrack G_t | S_t = s \rbrack \\
#   &= \mathbb{E}_{\pi} \lbrack
#      \sum_{k=0}^{\infty} \gamma ^ k R_{t+k+1} | S_t = s \rbrack \\
#   &= \mathbb{E}_{\pi} \lbrack
#      R_{t+1} + \gamma
#      \sum_{k=0}^{\infty} \gamma ^ k R_{t+k+2} | S_t = s \rbrack \\
#   &= \sum_a \pi(a|s) \sum_{s'} p(s' |s, a) \lbrack
#      r(s, a, s') + \gamma \mathbb{E}_{\pi} \lbrack
#          \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+2} | S_{t+1} = s'
#        \rbrack
#      \rbrack \\
#   &= \sum_a \pi(a|s) \sum_{s'} p(s' |s, a) \lbrack
#      r(s, a, s') + \gamma v_{\pi}(s')
#      \rbrack
# \end{align*}

# <markdowncell>

# Let us compute and visualize the state-value function through GridWorld
# example.
# For the detail of the definition, please refer to the Example 3.8
# from the book.
# - Agent moves in 5 x 5 grid cell
# - Agent takes action to move either north, east, west, or south.
# - Actions that would move the agent out of grid results in reward of -1
# and agent stays where it was before taking the action.
# - On success full action, agent moves to new position and receives
# 0 reward.
# - In special state $A$ (0, 1), all action cause the agent to move to $A'$
# (4, 1) and reward of 10
# - Similarly in $B$ (0, 3), all action cause the agent to move to $B'$
# (2, 3) and reward of 5

# First, we define `GridWorld` environment as follow

# <codecell>
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import luchador.env
import luchador.agent
from luchador.episode_runner import EpisodeRunner


def _transit(position, action):
    """Transition rule of GridWorld

    Parameters
    ----------
    position : NumPy NDArray
        Coordinate of agent
    action : int
        0, 1, 2, or 3, meaning north, east, west or south respectively

    Returns
    -------
    NumPy NDArray
        New coordinate
    int
        Reward
    """
    reward = 0
    new_position = position.copy()

    if np.all(position == [0, 1]):
        reward = 10
        new_position[:] = [4, 1]
        return new_position, reward
    if np.all(position == [0, 3]):
        reward = 5
        new_position[:] = [2, 3]
        return new_position, reward

    if action == 0:  # North
        move = [-1, 0]
    elif action == 1:  # East
        move = [0, 1]
    elif action == 2:  # West
        move = [0, -1]
    elif action == 3:  # South
        move = [1, 0]

    new_position = new_position + move
    if np.any(new_position < 0) or np.any(new_position > 4):
        reward = -1
        new_position[new_position < 0] = 0
        new_position[new_position > 4] = 4
    return new_position, reward


class GridWorld(luchador.env.BaseEnvironment):
    """GridWorld Example from Sutton, Chapter3."""
    def __init__(self, seed=None):
        self.position = None
        self.rng = np.random.RandomState(seed=seed)

    @property
    def n_actions(self):
        return 4

    def reset(self):
        """Reset position randomly"""
        self.position = self.rng.randint(5, size=2)
        return luchador.env.Outcome(
            reward=0, observation=self.position, terminal=False, state={})

    def step(self, action):
        """Move position based on action and transit rule"""
        self.position, reward = _transit(self.position, action)
        return luchador.env.Outcome(
            reward=reward, observation=self.position, terminal=False, state={})


# <markdowncell>

# Then we create agent which
# - has equiprobable random policy
# - estimates of action-value function via Monte-Calro approach

# <codecell>

class GridWorldAgent(luchador.agent.BaseAgent):
    """Agent walk on GridWorld with equiprobable random policy while
    estimating the action values

    Parameters
    ----------
    step_size : float
        StepSize parameter for estimating action value function
    discount : float
        Discount rate for computing state-value function
    initial_q : float
        Initial action value estimation
    """
    def __init__(self, step_size=0.9, discount=0.9, initial_q=10):
        self.step_size = step_size
        self.discount = discount

        self.position = None  # Pre-action position
        self.action_values = initial_q * np.ones((5, 5, 4))

    @property
    def state_values(self):
        """Current estimated state value mapping"""
        return np.mean(self.action_values, axis=2)

    def init(self, _):
        pass

    def reset(self, observation):
        self.position = observation

    def observe(self, action, outcome):
        pos0, pos1 = self.position, outcome.observation

        post_state_value = self.state_values[pos1[0], pos1[1]]
        target = outcome.reward + self.discount * post_state_value

        self.action_values[pos0[0], pos0[1], action] += self.step_size * (
            target - self.action_values[pos0[0], pos0[1], action])
        self.position = pos1

    def act(self):
        return np.random.choice(4)


# <markdowncell>

# We run the agent in the environment for setps long enough for action value
# estimation to get close enough to theoritical value as given in the book,
# Fig.3.5.

# <codecell>

def run_agent(env, agent, episodes=1000, steps=4):
    """Run agent for the given steps and plot the resulting state value"""
    runner = EpisodeRunner(env, agent, max_steps=steps)

    for _ in range(episodes):
        runner.run_episode()

    print('State Value:\n', agent.state_values)
    for i, action in enumerate(['north', 'east', 'west', 'south']):
        print('Action Value:', action)
        print(agent.action_values[:, :, i])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = ax.imshow(agent.state_values, interpolation='nearest', origin='upper')
    fig.colorbar(img)
    plt.show(block=False)


run_agent(
    env=GridWorld(seed=0),
    agent=GridWorldAgent(step_size=0.9, discount=0.9),
    steps=5, episodes=2000,
)


# <markdowncell>

# Let's compute the state values for optimal policy. $\pi^{*}$
# Computing optimal policy using Monte Carlo (sampling) approach is not
# straight forward, as agent has to explore the transitions which are not
# optimal.
# To overcome this, we take advantage of
# - random initialization: All states are visited eventually
# - Optimal initial value: All actions are tried eventually

# <codecell>


class GreedyGridWorldAgent(GridWorldAgent):
    """Agent act on greedy policy"""
    def __init__(self, step_size=0.9, discount=0.9, initial_q=30):
        super(GreedyGridWorldAgent, self).__init__(
            step_size=step_size, discount=discount, initial_q=initial_q)

    @property
    def state_values(self):
        return np.max(self.action_values, axis=2)

    def act(self):
        return np.argmax(
            self.action_values[self.position[0], self.position[1]])


run_agent(
    env=GridWorld(seed=0),
    agent=GreedyGridWorldAgent(step_size=0.9, discount=0.9, initial_q=30),
    steps=100, episodes=100,
)
